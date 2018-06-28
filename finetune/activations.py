"""
Activation functions
"""
import math

import tensorflow as tf



def gelu(x):
    return 0.5 * x * (1 + tf.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))


def swish(x):
    return x * tf.nn.sigmoid(x)


act_fns = {
    'relu':tf.nn.relu,
    'swish':swish,
    'gelu':gelu
}
