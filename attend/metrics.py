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


def streaming_reduce_sum(t, axis=None, weights=None, scope=None):
    if not axis is None and not type(axis) in [tuple, list]:
        axis = [axis]

    with tf.variable_scope(scope, 'streaming_reduce_sum', reuse=False):
        if not axis is None:
            keep_axis = np.delete(np.arange(t.shape.ndims), axis)
            dims = tf.shape(t)
            out_shape = tf.gather(dims, keep_axis)
            out_dims = np.delete(np.array(t.shape), axis) # tf.Dimension
            # reduce_dims = tf.gather(dims, axis)
        else:
            out_shape = []
            out_dims = []

        total = tf.get_variable('total', trainable=False,
                                initializer=tf.zeros(out_shape),
                                collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                validate_shape=False)
        total.set_shape(out_dims)
        total = tf.assign(total, total + tf.reduce_sum(t, axis))

        return total



def streaming_reduce_mean(t, axis=None, weights=None, scope=None):
    """
    Axis can be an array of axes
    """

    if axis is None:
        return tf.contrib.metrics.streaming_mean(t, weights)

    if not type(axis) in [tuple, list]:
        axis = [axis]

    if weights is None:
        weights = tf.ones_like(t)

    weights.shape.assert_is_compatible_with(t.shape)
    weights = tf.cast(weights, tf.float32)

    with tf.variable_scope(scope, 'streaming_reduce_mean'):
        keep_axis = np.delete(np.arange(t.shape.ndims), axis)
        dims = tf.shape(t)
        out_shape = tf.gather(dims, keep_axis)
        out_dims = np.delete(np.array(t.shape), axis) # tf.Dimension
        # reduce_dims = tf.gather(dims, axis)

        # reduced_count = tf.cast(tf.reduce_prod(reduce_dims), tf.float32)
        reduced_count = tf.reduce_sum(weights, axis)
        t = tf.multiply(t, weights)

        total = streaming_reduce_sum(t, axis)
        count = streaming_reduce_sum(weights, axis)

        # Put 1's where 0's would have been, otherwise it'd be 0/0
        count_ones_for_zeros = tf.cast(tf.logical_not(tf.cast(count, tf.bool)), tf.float32)

        return tf.divide(total, count + count_ones_for_zeros)


def streaming_icc(case, typ):
    assert typ == 1, 'Only ICC(?,1) supported (type 1)'

    def _icc(labels, predictions, weights=None, scope=None):
        labels.shape.assert_is_compatible_with(predictions.shape)
        if labels.shape.ndims == 2:
            # Assume the trailing 1 is forgotten
            labels = tf.expand_dims(labels, -1)
            predictions = tf.expand_dims(predictions, -1)
            if not weights is None:
                weights = tf.expand_dims(weights, -1)

        labels.shape.assert_has_rank(3)

        with tf.variable_scope(scope, 'icc_{}_{}'.format(case, typ)):
            shape = tf.shape(labels)
            B, T, r = tf.unstack(shape)
            n = B * T
            # NOTE alright change of heart, do away with batch dimension
            # B x T x 1 => 1 x B.T x 1 (to keep the old batched code)
            y_hat = tf.reshape(predictions, [1, n, r])
            y_lab = tf.reshape(labels, [1, n, r])
            weights = tf.reshape(weights, [1, n, r])


            assert_r = tf.assert_equal(r, 1, message='I dont think this operation is valid for r > 1')
            with tf.control_dependencies([assert_r]):
                total_n = streaming_reduce_sum(weights)
                total_n = tf.cast(total_n, tf.float32)

            # B x 2 x T x 1
            Y = tf.stack([y_hat, y_lab], axis=1)

            # Number of ratings, should be 1 for my case
            # k = Y.shape[3]
            k = 2 # Robert seems to set this to 2 and it makes sense
            # I am currently convinced k=2 because there is 1 truth and 1 prediction,
            # per target, that is
            k = tf.cast(Y.shape[1].value, tf.float32)

            # mean per target
            mpt = streaming_reduce_mean(Y, 1,
                                        tf.stack([weights,weights], 1)),   # B x T x 1

            tm = streaming_reduce_mean(mpt, 1) # B x 1

            # mean per rating
            mpr = streaming_reduce_mean(Y, 2) # B x 2 x 1

            # within target sum sqrs
            # The weights multiply makes sure 0-weighted entries don't count
            WSS = streaming_reduce_sum(
                tf.multiply(tf.square(Y[:,0] - mpt), weights) +
                tf.multiply(tf.square(Y[:,1] - mpt), weights), 1) # B x 1

            # within mean sqrs
            WMS = WSS / total_n / (k - 1) # B x 1

            # Between rater sum sqrs
            RSS = streaming_reduce_sum(
                tf.multiply(tf.square(mpr-tf.expand_dims(tm, 1), weights),
                            1)) * total_n # B x 1

            # Between rater mean sqrs
            RMS = RSS / (k - 1) # B x 1

            # Between target sum sqrs
            BSS = streaming_reduce_sum(
                tf.multiply(tf.square(mpt-tf.expand_dims(tm, 1), weights),
                            1)) * total_n # B x 1

            # Between target mean squares
            BMS = BSS / (total_n - 1) # B x 1

            # Residual sum of squares
            ESS = WSS - RSS # B x 1

            # Residual mean squares
            EMS = ESS / (total_n - 1) / (k - 1)

            if case == 3:
                if typ == 1:
                    res = (BMS - EMS) / (BMS + (k - 1) * EMS)

            res # B x 1

        r = labels.shape.as_list()[-1]
        if r == 1:
            icc_score = tf.reshape(res, [])
        else:
            icc_score = tf.reshape(res, [r])

        # Since ICC is in [-1, 1], and 1 is the best,
        # transform the same interval to [2, 0]
        icc_loss = 2 - (icc_score + 1)

        return icc_loss

    return _icc
