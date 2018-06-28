import os
import sys
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.python.client import device_lib
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from finetune import config


def format_gpu_string(num):
    return '/device:GPU:{}'.format(num)


def get_available_gpus():
    if config.VISIBLE_GPUS is not None:
        return config.VISIBLE_GPUS
    local_device_protos = device_lib.list_local_devices()
    config.VISIBLE_GPUS = [
        int(x.name.split(':')[-1]) for x in local_device_protos
        if x.device_type == 'GPU'
    ]
    return config.VISIBLE_GPUS


def shape_list(x):
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


def np_softmax(x, t=1):
    x = x / t
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)


def make_path(f):
    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        os.makedirs(d)
    return f


def _identity_init(shape, dtype, partition_info, scale):
    n = shape[-1]
    w = np.eye(n) * scale
    if len([s for s in shape if s != 1]) == 2:
        w = w.reshape(shape)
    return w.astype(np.float32)


def identity_init(scale=1.0):
    return partial(_identity_init, scale=scale)


def _np_init(shape, dtype, partition_info, w):
    return w


def np_init(w):
    return partial(_np_init, w=w)


def find_trainable_variables(key, exclude=None):
    trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ".*{}.*".format(key))
    if exclude is not None:
        trainable_variables = [
            var for var in trainable_variables
            if exclude not in var.name
        ]
    return trainable_variables


def soft_split(*xs, n_splits=None):
    """
    Similar to tf.split but can accomodate batches that are not evenly divisible by n_splits
    """
    if not n_splits or not isinstance(n_splits, int):
        raise ValueError("n_splits must be a valid integer.")

    x = xs[0]
    current_batch_size = tf.shape(x)[0]
    n_per = tf.to_int32(tf.ceil(current_batch_size / n_splits))
    for i in range(n_splits):
        start = i * n_per
        end = tf.minimum((i + 1) * n_per, current_batch_size)
        i_range = tf.range(start, end)
        yield [tf.gather(x, i_range) for x in xs]


def flatten(outer):
    return [el for inner in outer for el in inner]


def remove_none(l):
    return [e for e in l if e is not None]


def iter_data(*datas, n_batch=128, truncate=False, verbose=False, max_batches=float("inf")):
    n = len(datas[0])
    if truncate:
        n = (n // n_batch) * n_batch
    n = min(n, max_batches * n_batch)
    n_batches = 0
    if verbose:
        f = sys.stderr
    else:
        f = open(os.devnull, 'w')
    for i in tqdm(range(0, n, n_batch), total=n // n_batch, file=f, ncols=80, leave=False):
        if n_batches >= max_batches: raise StopIteration
        if len(datas) == 1:
            yield datas[0][i:i + n_batch]
        else:
            yield (d[i:i + n_batch] for d in datas)
        n_batches += 1


def get_ema_if_exists(v, gvs):
    name = v.name.split(':')[0]
    ema_name = name + '/ExponentialMovingAverage:0'
    ema_v = [v for v in gvs if v.name == ema_name]
    if len(ema_v) == 0:
        ema_v = [v]
    return ema_v[0]


def get_ema_vars(*vs):
    if tf.get_variable_scope().reuse:
        gvs = tf.global_variables()
        vs = [get_ema_if_exists(v, gvs) for v in vs]
    if len(vs) == 1:
        return vs[0]
    else:
        return vs


@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
    """force gradient to be a dense tensor
    it's often faster to do dense embedding gradient on GPU than sparse on CPU
    """
    return x


def assign_to_gpu(gpu=0, params_device="/device:CPU:0"):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op == "Variable":
            return params_device
        else:
            return "/gpu:%d" % gpu

    return _assign


def average_grads(tower_grads):
    def average_dense(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        grad = grad_and_vars[0][0]
        for g, _ in grad_and_vars[1:]:
            grad += g
        return grad / len(grad_and_vars)

    def average_sparse(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        indices = []
        values = []
        for g, _ in grad_and_vars:
            indices += [g.indices]
            values += [g.values]
        indices = tf.concat(indices, 0)
        values = tf.concat(values, 0)
        return tf.IndexedSlices(values, indices, grad_and_vars[0][0].dense_shape)

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        if grad_and_vars[0][0] is None:
            grad = None
        elif isinstance(grad_and_vars[0][0], tf.IndexedSlices):
            grad = average_sparse(grad_and_vars)
        else:
            grad = average_dense(grad_and_vars)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


class OrdinalClassificationEncoder:
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val
        self.lookup = None
        self.inverse_lookup = None
        self.keys = None
        self.classes_ = [0, 1]

    def fit(self, y):
        self.keys = list(set(y))
        self.keys.sort()
        spaced_probs = np.linspace(self.min_val, self.max_val, len(self.keys))
        prob_distributions = np.transpose([spaced_probs, 1 - spaced_probs])
        self.inverse_lookup = spaced_probs
        self.lookup = dict(zip(self.keys, prob_distributions))

    def transform(self, y):
        return list(map(self.lookup.get, y))

    def inverse_transform(self, y):
        output = []
        for item in y:
            i_min = np.argmin(np.abs(self.inverse_lookup - item[0]))
            output.append(self.keys[i_min])
        return np.asarray(output)

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)


class OneHotLabelEncoder(LabelEncoder):

    def _make_one_hot(self, labels):
        output = np.zeros([len(labels), len(self.classes_)], dtype=np.float)
        output[np.arange(len(labels)), labels] = 1
        return output

    def fit_transform(self, y):
        labels = super().fit_transform(y)
        return self._make_one_hot(labels)

    def transform(self, y):
        labels = super().transform(y)
        return self._make_one_hot(labels)


