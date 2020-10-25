import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def OptJack(x, y):
    y = K.cast(y[:,:,:] > 0.5, 'float32')
    k = tf.constant(np.ones((3,3)).reshape((3,3,1,1)), dtype=tf.float32)
    convedX = tf.nn.conv2d(x, k, strides=[1,1,1,1], padding="SAME")
    convedY = tf.nn.conv2d(y, k, strides=[1,1,1,1], padding="SAME")
    m = tf.constant([1.0], dtype=tf.float32)
    convedX = tf.minimum(convedX, m)
    convedY = tf.minimum(convedY, m)
    x = tf.math.equal(x, m)
    convedX = tf.math.equal(convedX, m)
    convedY = tf.math.equal(convedY, m)
    y = tf.math.equal(y, m)
    o = tf.logical_or(x,y)
    a1 = tf.logical_and(x, convedY)
    a2 = tf.logical_and(convedX, y)
    o = tf.math.count_nonzero(o, dtype="float32")
    a1 = tf.math.count_nonzero(a1)
    a2 = tf.math.count_nonzero(a2)
    a = K.cast(a1+a2, "float32") / 2.0
    iou = tf.where(tf.equal(o, 0), 1., tf.cast(a/o, 'float32'))
    return iou
