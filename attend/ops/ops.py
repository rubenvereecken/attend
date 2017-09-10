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

    # Put 1's where 0's would have been, otherwise it'd be 0/0
    # It works because total will be 0 in those locations anyway
    count = tf.reduce_sum(weights, axis)
    count_ones_for_zeros = tf.cast(tf.logical_not(tf.cast(count, tf.bool)), tf.float32)
    count = count + count_ones_for_zeros

    return tf.divide(total, count)
