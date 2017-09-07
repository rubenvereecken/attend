import tensorflow as tf


def repeat(x, n, name=None):
    with tf.name_scope(name or 'repeat'):
        l = x.shape.ndims
        x = tf.expand_dims(x, 0)
        m = tf.concat([[n], tf.ones(l, dtype=tf.int32)], axis=0)
        x = tf.tile(x, m)
        return x


def reduce_average(x, axis, weights):
    x.shape.assert_is_compatible_with(weights.shape)
    x = tf.multiply(x, weights)
    total = tf.reduce_sum(x, axis)
    count = tf.reduce_sum(weights, axis)

    return tf.divide(total, count)
