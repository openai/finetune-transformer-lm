import os
import random

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

from finetune.encoding import TextEncoder
from finetune.optimizers import AdamWeightDecay
from finetune.utils import find_trainable_variables, shape_list
from finetune.config import MAX_LENGTH, BATCH_SIZE, WEIGHT_STDDEV, N_EPOCHS, CLF_P_DROP, SEED

SHAPES_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'params_shapes.json')
PARAM_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'params_{}.npy')
N_EMBED = 768



def clf(x, ny, w_init=tf.random_normal_initializer(stddev=WEIGHT_STDDEV), b_init=tf.constant_initializer(0), train=False):
    with tf.variable_scope('clf'):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [nx, ny], initializer=w_init)
        b = tf.get_variable("b", [ny], initializer=b_init)
        return tf.matmul(x, w) + b

def model(X, M, Y, n_classes, train=False, reuse=False, max_length=MAX_LENGTH):
    with tf.variable_scope('model', reuse=reuse):
        we = tf.get_variable("we", [encoder.vocab_size + max_length, N_EMBED], initializer=tf.random_normal_initializer(stddev=WEIGTH_STDDEV))
        we = dropout(we, embd_pdrop, train)

        X = tf.reshape(X, [-1, max_length, 2])
        M = tf.reshape(M, [-1, max_length])

        h = embed(X, we)
        for layer in range(n_layer):
            h = block(h, 'h%d'%layer, train=train, scale=True)

        # language model ignores last hidden state because we don't have a target
        lm_h = tf.reshape(h[:, :-1], [-1, N_EMBED]) # [batch, seq_len, embed] --> [batch * seq_len, embed]
        lm_logits = tf.matmul(lm_h, we, transpose_b=True) # tied weights
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=lm_logits,
            labels=tf.reshape(X[:, 1:, 0], [-1])
        )
        lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1] - 1])
        lm_losses = tf.reduce_sum(lm_losses * M, 1) / tf.reduce_sum(M, 1)

        # Use hidden state at classifier token as input to final proj. + softmax
        clf_h = tf.reshape(h, [-1, N_EMBED]) # [batch * seq_len, embed]
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1), tf.int32)
        clf_h = tf.gather(clf_h, tf.range(shape_list(X)[0], dtype=tf.int32) * max_length + pool_idx)

        clf_h = tf.reshape(clf_h, [-1, N_EMBED])
        if train and clf_pdrop > 0:
            clf_h = tf.nn.dropout(clf_h, 1 - CLF_P_DROP)
        clf_logits = clf(clf_h, n_classes, train=train)
        # clf_logits = tf.reshape(clf_logits, [-1, n_classes])

        clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits, labels=Y)
        return clf_logits, clf_losses, lm_losses


class LanguageModelClassifier(object):

    def __init__(self, max_length=MAX_LENGTH, *args, **kwargs):
        # ensure results are reproducible
        random.seed(SEED)
        np.random.seed(SEED)
        tf.set_random_seed(SEED)

        self.encoder = TextEncoder()

        # tf placeholders
        self.X = tf.placeholder(tf.int32,   [None, max_length, 2]) # token idxs (BPE embedding + positional)
        self.M = tf.placeholder(tf.float32, [None, max_length])    # sequence mask
        self.Y = tf.placeholder(tf.int32,   [None])                # classification targets

        # symbolic ops
        self.logits    = None # classification logits
        self.clf_loss  = None # cross-entropy loss
        self.lm_losses = None # language modeling losses
        self.train     = None # gradient + parameter update

    def finetune(self, X, Y, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
        """
        X: List / array of text
        Y: Class labels
        """
        token_idxs = self.encoder.encode_for_classification(X, max_length=max_length)
        train_x, train_mask = self._array_format(token_idxs)
        n_classes = len(np.unique(Y))
        self._build_model(self.X, self.M, self.Y, n_classes=n_classes)
        self._load_saved_params()

        dataset = shuffle(train_x, train_mask, Y, random_state=np.random)

        best_score = 0
        for i in range(N_EPOCHS):
            N_BATCH_TRAIN = BATCH_SIZE * N_GPUS
            for xmb, mmb, ymb in iter_data(*dataset, n_batch=N_BATCH_TRAIN, truncate=True, verbose=True):
                cost, _ = sess.run([self.clf_loss, self.train], {X: xmb, M: mmb, Y: ymb})
                n_updates += 1
            n_epochs += 1


    def _array_format(self, token_idxs, max_length=MAX_LENGTH):
        """
        Returns numpy array of token idxs and corresponding mask
        Returned `x` array contains two channels:
            0: byte-pair encoding embedding
            1: positional embedding
        """
        n = len(token_idxs)
        seq_lengths = [len(x) for x in token_idxs]
        x    = np.zeros((n, max_length, 2), dtype=np.int32)
        mask = np.zeros((n, max_length), dtype=np.float32)
        for i, seq_length in enumerate(seq_lengths):
            # BPE embedding
            x[:, :seq_length, 0] = token_idxs[i]
            # masking: value of 1 means "consider this in cross-entropy LM loss"
            mask[:, 1:seq_length] = 1
        # positional_embeddings
        x[:, :, 1] = np.arange(encoder.vocab_size, encoder.vocab_size + max_length)
        return x, mask

    def _build_model(self, X, M, Y):
        """
        Finetune language model on text inputs
        """
        gpu_ops = []
        gpu_grads = []
        X = tf.split(X, n_gpu, 0)
        M = tf.split(M, n_gpu, 0)
        Y = tf.split(Y, n_gpu, 0)
        for i, (splitX, splitM, splitY) in enumerate(zip(X, M, Y)):
            do_reuse = True if i > 0 else None
            device = tf.device(assign_to_gpu(i, "/gpu:0"))
            scope = tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse)
            with device, scope:
                clf_logits, clf_losses, lm_losses = model(splitX, splitM, splitY, train=True, reuse=do_reuse)
                if lm_coef > 0:
                    train_loss = tf.reduce_mean(clf_losses) + lm_coef * tf.reduce_mean(lm_losses)
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
            lr=lr,
            schedule=partial(lr_schedules[lr_schedule], warmup=lr_warmup),
            t_total=n_updates_total,
            l2=l2,
            max_grad_norm=max_grad_norm,
            vector_l2=vector_l2,
            b1=b1,
            b2=b2,
            e=e
        )
        self.clf_loss = tf.reduce_mean(self.clf_losses)

    def _load_saved_params(self, max_length=MAX_LENGTH):
        """
        Load serialized model parameters into tf Tensors
        """
        pretrained_params = find_trainable_variables('model', exclude='model/clf')
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(tf.initialize_global_variables())

        shapes = json.load(open('model/params_shapes.json'))
        offsets = np.cumsum([np.prod(shape) for shape in shapes])
        init_params = [np.load('model/params_{}.npy'.format(n)) for n in range(10)]
        init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
        init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
        init_params[0] = init_params[0][:max_length]
        init_params[0] = np.concatenate([init_params[1], (np.random.randn(len(encoder.special_tokens), N_EMBED) * WEIGTH_STDDEV).astype(np.float32), init_params[0]], 0)
        del init_params[1]
        sess.run([p.assign(ip) for p, ip in zip(pretrained_params, init_params)])


if __name__ == "__main__":
    df = pd.read_csv("data/AirlineNegativity.csv")
    model = LanguageModelClassifier()
    model.finetune(df.Text.values, df.Target.values)
