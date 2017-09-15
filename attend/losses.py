import tensorflow as tf
# import attend.ops as ops
from attend import ops

def root_mean_squared_error(labels, predictions, weights=1., scope=None,
                            loss_collection=tf.GraphKeys.LOSSES):
    with tf.name_scope(scope, 'root_mean_squared_error'):
        mse = tf.losses.mean_squared_error(labels, predictions, weights,
                                           loss_collection=None)
        rmse = tf.sqrt(mse)

        if loss_collection:
            tf.add_to_collection(loss_collection, rmse)

    return rmse


def icc(case, typ, weighted=True):
    """
    Functionally equivalent to Robert's

    Assumes data of shape
        B x n x r
        where r is the number of ratings, usually 1
        and n is the amount of targets, useful for sequences
    """
    assert typ == 1, 'Only ICC(?,1) supported (type 1)'

    def _icc(labels, predictions, weights=None, scope=None,
             loss_collection=tf.GraphKeys.LOSSES):
        labels.shape.assert_is_compatible_with(predictions.shape)
        if labels.shape.ndims == 2:
            labels = tf.expand_dims(labels, -1)
        if predictions.shape.ndims == 2:
            predictions = tf.expand_dims(predictions, -1)
        if not weights is None and weights.shape.ndims == 2:
            weights = tf.expand_dims(weights, -1)

        if weights is None:
            weights = tf.ones_like(predictions)
        if not weights.dtype.is_compatible_with(tf.float32):
            weights = tf.cast(weights, tf.float32)

        with tf.variable_scope(scope, 'icc_{}_{}'.format(case, typ)):
            shape = tf.shape(labels)
            B, T, r = tf.unstack(shape)
            n = B * T

            # Important to use 'total_n' as it is a masked/weighted count
            assert_single_target = tf.assert_equal(r, 1, message='Only r=1 supported')
            with tf.control_dependencies([assert_single_target]):
                total_n = tf.cast(tf.reduce_sum(weights), tf.float32)

            # NOTE alright change of heart, do away with batch dimension
            # B x T x 1 => 1 x B.T x 1 (to keep the old batched code)
            y_hat = tf.reshape(predictions, [1, n, r])
            y_lab = tf.reshape(labels, [1, n, r])
            weights = tf.reshape(weights, [1, n, r])
            y_hat = tf.multiply(y_hat, weights)
            y_lab = tf.multiply(y_lab, weights)

            # B x 2 x T x 1
            Y = tf.stack([y_hat, y_lab], axis=1)
            Y_weights = tf.stack([weights, weights], 1)

            # Number of ratings, should be 1 for my case
            # k = Y.shape[3]
            k = 2 # Robert seems to set this to 2 and it makes sense
            # I am currently convinced k=2 because there is 1 truth and 1 prediction,
            # per target, that is
            k = tf.cast(Y.shape[1].value, tf.float32)

            # number of targets
            # n = Y.shape[2].value # I think this is T
            n = tf.cast(n, tf.float32)

            # mean per target
            mpt = ops.reduce_average(Y, 1, Y_weights)  # B x T x 1
            if not weighted:
                mpt = tf.reduce_mean(Y, 1)

            # tm = tf.reduce_mean(mpt, 1) # B x 1
            tm = ops.reduce_average(Y, [1,2], Y_weights)
            if not weighted:
                tm = tf.reduce_mean(Y, [1,2])

            # mean per rating
            mpr = ops.reduce_average(Y, 2, Y_weights) # B x 2 x 1
            if not weighted:
                mpr = tf.reduce_mean(Y, 2)

            # within target sum sqrs
            WSS = tf.reduce_sum(
                tf.square(Y[:,0] - mpt) +
                tf.square(Y[:,1] - mpt), 1) # B x 1

            # within mean sqrs
            WMS = WSS / total_n / (k - 1) # B x 1

            # Between rater sum sqrs
            rss_tmp = tf.square(mpr-tf.expand_dims(tm, 1))
            RSS = tf.reduce_sum(rss_tmp, 1) * total_n # B x 1

            # Between rater mean sqrs
            RMS = RSS / (k - 1) # B x 1

            # Between target sum sqrs
            BSS = tf.reduce_sum(tf.square(mpt-tf.expand_dims(tm, 1)), 1) * k # B x 1

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

        return icc_score

    return _icc


def icc_loss(case, typ, weighted=True):
    icc_fun = icc(case, typ, weighted)

    def icc_loss_wrapper(*args, **kwargs):
        icc_score = icc_fun(*args, **kwargs)

        # Since ICC is in [-1, 1], and 1 is the best,
        # transform the same interval to [2, 0]
        icc_loss = 2 - (icc_score + 1)

        return icc_loss

    return icc_loss_wrapper
