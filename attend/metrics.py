import tensorflow as tf
import numpy as np

def streaming_mse(predictions, targets, keys, lengths, mask=None):
    with tf.name_scope('streaming_mse'):
        totals = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string,
                    value_dtype=tf.float32, default_value=0,
                    name='total_table')
        counts = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string,
                    value_dtype=tf.float32, default_value=0,
                    name='count_table')
        mse = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string,
                    value_dtype=tf.float32, default_value=0,
                    name='mse_table')

        total_se = totals.lookup(keys)
        masked_diff = (predictions - targets) * mask
        current_se = tf.reduce_sum(tf.square(masked_diff, name='mse'), axis=1)
        total_se += current_se
        total_update = totals.insert(keys, total_se)
        count = counts.lookup(keys)
        count += tf.cast(lengths, tf.float32)
        count_update = counts.insert(keys, count)
        mse_update = mse.insert(keys, total_se / (count))

    with tf.control_dependencies([total_update, count_update, mse_update]):
        return mse.lookup(keys), mse


def streaming_reduce_mean(t, axis=None, weights=None, scope=None):
    """
    Axis can be an array of axes
    """

    if axis is None:
        return tf.contrib.metrics.streaming_mean(t, weights)

    if not type(axis) in [tuple, list]:
        axis = [axis]


    with tf.variable_scope(scope, 'streaming_reduce_mean'):
        keep_axis = np.delete(np.arange(t.shape.ndims), axis)
        dims = tf.shape(t)
        out_shape = tf.gather(dims, keep_axis)
        out_dims = np.delete(np.array(t.shape), axis) # tf.Dimension
        reduce_dims = tf.gather(dims, axis)

        total = tf.get_variable('total', trainable=False,
                                initializer=tf.zeros(out_shape),
                                collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                validate_shape=False)
        count = tf.get_variable('count', trainable=False,
                                initializer=0.,
                                collections=[tf.GraphKeys.LOCAL_VARIABLES])
        total.set_shape(out_dims)
        reduced_count = tf.cast(tf.reduce_prod(reduce_dims), tf.float32)

        total = tf.assign(total, total + tf.reduce_sum(t, axis))
        count = tf.assign(count, count + reduced_count)

        return total / count


def streaming_icc(predictions, targets, lengths, mask=None,
                  cse=3, typ=1):
    """
    'cse' is either 1,2,3. 'cse' is: 1 if each target is measured by a
    different set of raters from a population of raters, 2 if each target is
    measured by the same raters, but that these raters are sampled from a
    population of raters, 3 if each target is measured by the same raters and
    these raters are the only raters of interest.

    'typ' is either 'single' or 'k' & denotes whether the ICC is based on a
    single measurement or on an average of k measurements, where k = the
    number of ratings/raters.

    ONLY EXPECT THIS TO WORK FOR r,1! I.e single observation
    """
    # TODO rm Robert adds a leading 1-dimension, so I think it's batched
    # B x T x 1
    y_hat = predictions
    y_lab = targets

    Y = tf.stack([y_hat, y_lab])

    # number of targets
    n = Y.shape[2]

    # mean per target
    mpt = tf.contrib.metrics.streaming_mean()

