import os
import random
import json
import pickle

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

from functools import partial
from finetune.encoding import TextEncoder
from finetune.optimizers import AdamWeightDecay
from finetune.utils import find_trainable_variables, shape_list, assign_to_gpu, average_grads, iter_data
from finetune.config import MAX_LENGTH, BATCH_SIZE, WEIGHT_STDDEV, N_EPOCHS, CLF_P_DROP, SEED, N_GPUS, WEIGHT_STDDEV, EMBED_P_DROP, RESID_P_DROP, N_HEADS, N_LAYER, ATTN_P_DROP, ACT_FN, LM_LOSS_COEF, LR, B1, B2, L2_REG, VECTOR_L2, EPSILON,LR_SCHEDULE, MAX_GRAD_NORM, LM_LOSS_COEF, LR_WARMUP
from finetune.train import block, dropout, embed, lr_schedules

SHAPES_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'params_shapes.json')
PARAM_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'params_{}.npy')
N_EMBED = 768


def clf(x, ny, w_init=tf.random_normal_initializer(stddev=WEIGHT_STDDEV), b_init=tf.constant_initializer(0), train=False):
    with tf.variable_scope('clf'):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [nx, ny], initializer=w_init)
        b = tf.get_variable("b", [ny], initializer=b_init)
        return tf.matmul(x, w) + b

def model(X, M, Y, n_classes, encoder, train=False, reuse=False, max_length=MAX_LENGTH):
    with tf.variable_scope('model', reuse=reuse):
        we = tf.get_variable("we", [encoder.vocab_size + max_length, N_EMBED], initializer=tf.random_normal_initializer(stddev=WEIGHT_STDDEV))
        we = dropout(we, EMBED_P_DROP, train)

        X = tf.reshape(X, [-1, max_length, 2])
        M = tf.reshape(M, [-1, max_length])

        h = embed(X, we)
        for layer in range(N_LAYER):
            h = block(h, N_HEADS, ACT_FN, RESID_P_DROP, ATTN_P_DROP,'h%d'%layer, train=train, scale=True)

        # language model ignores last hidden state because we don't have a target
        lm_h = tf.reshape(h[:, :-1], [-1, N_EMBED]) # [batch, seq_len, embed] --> [batch * seq_len, embed]
        lm_logits = tf.matmul(lm_h, we, transpose_b=True) # tied weights
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=lm_logits,
            labels=tf.reshape(X[:, 1:, 0], [-1])
        )

        lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1] - 1])
        lm_losses = tf.reduce_sum(lm_losses * M[:, 1:], 1) / tf.reduce_sum(M[:, 1:], 1)

        # Use hidden state at classifier token as input to final proj. + softmax
        clf_h = tf.reshape(h, [-1, N_EMBED]) # [batch * seq_len, embed]
        clf_token = encoder['_classify_']
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1), tf.int32)
        clf_h = tf.gather(clf_h, tf.range(shape_list(X)[0], dtype=tf.int32) * max_length + pool_idx)

        clf_h = tf.reshape(clf_h, [-1, N_EMBED])
        if train and CLF_P_DROP > 0:
            clf_h = tf.nn.dropout(clf_h, 1 - CLF_P_DROP)
        clf_logits = clf(clf_h, n_classes, train=train)

        clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits, labels=Y)
        return clf_logits, clf_losses, lm_losses


class LanguageModelClassifier(object):

    def __init__(self, max_length=MAX_LENGTH, *args, **kwargs):
        # ensure results are reproducible
        self.max_length = max_length
        self.label_encoder = LabelEncoder()
        self._initialize()

    def _initialize(self):
        self._set_random_seed(SEED)
        self.encoder = TextEncoder()

        # symbolic ops
        self.logits    = None # classification logits
        self.clf_loss  = None # cross-entropy loss
        self.lm_losses = None # language modeling losses
        self.train     = None # gradient + parameter update

    def finetune(self, X, Y, batch_size=BATCH_SIZE):
        """
        X: List / array of text
        Y: Class labels
        """
        token_idxs = self.encoder.encode_for_classification(X, max_length=self.max_length)
        train_x, train_mask = self._array_format(token_idxs)
        n_classes = len(np.unique(Y))
        n_batch_train = BATCH_SIZE * N_GPUS
        n_updates_total = (len(Y) // n_batch_train) * N_EPOCHS
        Y = self.label_encoder.fit_transform(Y)
        self._build_model(n_updates_total=n_updates_total, n_classes=n_classes)
        self._load_saved_params()

        dataset = shuffle(train_x, train_mask, Y, random_state=np.random)

        best_score = 0
        for i in range(N_EPOCHS):
            for xmb, mmb, ymb in iter_data(*dataset, n_batch=n_batch_train, truncate=True, verbose=True):
                cost, _ = self.sess.run([self.clf_loss, self.train], {self.X: xmb, self.M: mmb, self.Y: ymb})


    def _array_format(self, token_idxs):
        """
        Returns numpy array of token idxs and corresponding mask
        Returned `x` array contains two channels:
            0: byte-pair encoding embedding
            1: positional embedding
        """
        n = len(token_idxs)
        seq_lengths = [len(x) for x in token_idxs]
        x    = np.zeros((n, self.max_length, 2), dtype=np.int32)
        mask = np.zeros((n, self.max_length), dtype=np.float32)
        for i, seq_length in enumerate(seq_lengths):
            # BPE embedding
            x[:, :seq_length, 0] = token_idxs[i]
            # masking: value of 1 means "consider this in cross-entropy LM loss"
            mask[:, 1:seq_length] = 1
        # positional_embeddings
        x[:, :, 1] = np.arange(self.encoder.vocab_size, self.encoder.vocab_size + self.max_length)
        return x, mask

    def _build_model(self, n_updates_total, n_classes, train=True, reuse=None):
        """
        Finetune language model on text inputs
        """
        gpu_ops = []
        gpu_grads = []

        self._define_placeholders()

        splitX = tf.split(self.X, N_GPUS, 0)
        splitM = tf.split(self.M, N_GPUS, 0)
        splitY = tf.split(self.Y, N_GPUS, 0)
        for i, (X, M, Y) in enumerate(zip(splitX, splitM, splitY)):
            do_reuse = reuse or True if i > 0 else None
            device = tf.device(assign_to_gpu(i, "/gpu:0"))
            scope = tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse)
            with device, scope:
                clf_logits, clf_losses, lm_losses = model(X, M, Y, n_classes=n_classes, encoder=self.encoder, train=train, reuse=do_reuse)
                if LM_LOSS_COEF > 0:
                    train_loss = tf.reduce_mean(clf_losses) + LM_LOSS_COEF * tf.reduce_mean(lm_losses)
                else:
                    train_loss = tf.reduce_mean(clf_losses)
                params = find_trainable_variables("model")
                grads = tf.gradients(train_loss, params)
                grads = list(zip(grads, params))
                gpu_grads.append(grads)
                gpu_ops.append([clf_logits, clf_losses, lm_losses])

        self.logits, self.clf_losses, self.lm_losses = [tf.concat(op, 0) for op in zip(*gpu_ops)]
        grads = average_grads(gpu_grads)
        grads = [g for g, p in grads]
        self.train = AdamWeightDecay(
            params=params,
            grads=grads,
            lr=LR,
            schedule=partial(lr_schedules[LR_SCHEDULE], warmup=LR_WARMUP),
            t_total=n_updates_total,
            l2=L2_REG,
            max_grad_norm=MAX_GRAD_NORM,
            vector_l2=VECTOR_L2,
            b1=B1,
            b2=B2,
            e=EPSILON
        )
        self.clf_loss = tf.reduce_mean(self.clf_losses)

        # Optionally load saved model
        if hasattr(self, '_save_path'):
            self.sess = tf.Session()
            saver = tf.train.Saver()
            saver.restore(self.sess, self._save_path)
            del self._save_path

    def _set_random_seed(self, seed=SEED):
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

    def _define_placeholders(self):
        # tf placeholders
        self.X = tf.placeholder(tf.int32,   [None, self.max_length, 2]) # token idxs (BPE embedding + positional)
        self.M = tf.placeholder(tf.float32, [None, self.max_length])    # sequence mask
        self.Y = tf.placeholder(tf.int32,   [None])                     # classification targets

    def _load_saved_params(self):
        """
        Load serialized base model parameters into tf Tensors
        """
        pretrained_params = find_trainable_variables('model', exclude='model/clf')
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())

        shapes = json.load(open('model/params_shapes.json'))
        offsets = np.cumsum([np.prod(shape) for shape in shapes])
        init_params = [np.load('model/params_{}.npy'.format(n)) for n in range(10)]
        init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
        init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
        init_params[0] = init_params[0][:self.max_length]
        init_params[0] = np.concatenate([init_params[1], (np.random.randn(len(self.encoder.special_tokens), N_EMBED) * WEIGHT_STDDEV).astype(np.float32), init_params[0]], 0)
        del init_params[1]
        self.sess.run([p.assign(ip) for p, ip in zip(pretrained_params, init_params)])

    def __getstate__(self):
        """
        Leave serialization of all tf objects to tf
        """
        required_fields = ['label_encoder', 'max_length', '_save_path']
        serialized_state = {
            k: v for k, v in self.__dict__.items()
            if k in required_fields
        }
        return serialized_state

    def save(self, path):
        """
        Save in two steps:
            - Serialize tf graph to disk using tf.Saver
            - Serialize python model using pickle
        """
        self._save_path = path
        saver = tf.train.Saver()
        saver.save(self.sess, path)
        pickle.dump(self, open(self._save_path + '.pkl', 'wb'))
        del self._save_path

    @classmethod
    def load(cls, path):
        """
        Load in three steps:
            - Load pickled python object
            - Clear tf graph
            - Load serialized session using tf.train.Saver
        """
        if not path.endswith('.pkl'):
            path += '.pkl'
        model = pickle.load(open(path, 'rb'))
        model._initialize()
        tf.reset_default_graph()
        return model

if __name__ == "__main__":
    df = pd.read_csv("data/AirlineNegativity.csv")
    classifier = LanguageModelClassifier()
    n_train = 10
    classifier.finetune(df.Text.values[:n_train], df.Target.values[:n_train])
    classifier.save('saved-models/airline-negativity')
    classifier = LanguageModelClassifier.load('saved-models/airline-negativity')
    classifier.finetune(df.Text.values[:n_train], df.Target.values[:n_train])
