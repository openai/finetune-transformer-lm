import math
import numpy as np
import tensorflow as tf

def warmup_cosine(x, warmup=0.002):
    s = tf.cast(x <= warmup, tf.float32)
    return s*(x/warmup) + (1-s)*(0.5 * (1 + tf.cos(math.pi * x)))

def warmup_constant(x, warmup=0.002):
    s = tf.cast(x <= warmup, tf.float32)
    return s*(x/warmup) + (1-s)*1

def warmup_linear(x, warmup=0.002):
    s = tf.cast(x <= warmup, tf.float32)
    return (s*(x/warmup) + (1-s))*(1-x)

schedules = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
}

def adam(params, grads, lr, schedule, t_total, b1=0.9, b2=0.999, e=1e-8, l2=0, vector_l2=False, max_grad_norm=-1, **kwargs):
    """
    adam with weight decay fix
    """
    t = tf.Variable(0, dtype=tf.float32, trainable=False)
    tt = t+1
    updates = [t.assign(tt)]
    if max_grad_norm > 0:
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    for p, g in zip(params, grads):
        if p is None or g is None:
            print("can't train", p.name, g)
        else:
            if isinstance(g, tf.IndexedSlices):
                g = tf.convert_to_tensor(g)
            m = tf.Variable(p*0, dtype=tf.float32, trainable=False)
            v = tf.Variable(p*0, dtype=tf.float32, trainable=False)
            lrt = lr*tf.sqrt(1-b2**tt)/(1-b1**tt)
            lrt *= schedule(t/t_total)
            mt = b1*m + (1-b1)*g
            vt = b2*v + (1-b2)*g*g
            if (len(p.get_shape()) > 1 or vector_l2) and l2 > 0:
                pt = p - lrt * (mt / (tf.sqrt(vt) + e) + l2*p)
            else:
                pt = p - lrt * (mt / (tf.sqrt(vt) + e))
            updates.extend([m.assign(mt), v.assign(vt), p.assign(pt)])
    return tf.group(*updates)
