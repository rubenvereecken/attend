import tensorflow as tf

def root_mean_squared_error(labels, predictions, weights=1., scope=None,
                            loss_collection=tf.GraphKeys.LOSSES):
    with tf.name_scope(scope, 'root_mean_squared_error'):
        mse = tf.losses.mean_squared_error(labels, predictions, weights,
                                           loss_collection=None)
        rmse = tf.sqrt(mse)

        if loss_collection:
            tf.add_to_collection(loss_collection, rmse)

    return rmse


def icc(case, typ,
        labels, predictions, weights=1., scope=None,
        loss_collection=tf.GraphKeys.LOSSES):
    """
    Functionally equivalent to Robert's

    Assumes data of shape
        B x n x r
        where r is the number of ratings, usually 1
        and n is the amount of targets, useful for sequences
    """
    labels.shape.assert_has_rank(3)

    assert typ == 1, 'Only ICC(?,1) supported (type 1)'

    # B x T x 1
    y_hat = predictions
    y_lab = labels

    # B x 2 x T x 1
    Y = tf.stack([y_hat, y_lab], axis=1)

    # Number of ratings, should be 1 for my case
    # k = Y.shape[3]
    k = 2 # Robert seems to set this to 2 and it makes sense
    # I am currently convinced k=2 because there is 1 truth and 1 prediction,
    # per target, that is
    k = Y.shape[1].value

    # number of targets
    n = Y.shape[2].value # I think this is T

    # mean per target
    mpt = tf.reduce_mean(Y, 1)  # B x T x 1

    tm = tf.reduce_mean(mpt, 1) # B x 1

    # mean per rating
    mpr = tf.reduce_mean(Y, 2) # B x 2 x 1

    # within target sum sqrs
    WSS = tf.reduce_sum(
        tf.square(Y[:,0] - mpt) +
        tf.square(Y[:,1] - mpt), 1) # B x 1

    # within mean sqrs
    WMS = WSS / n / (k - 1) # B x 1

    # Between rater sum sqrs
    RSS = tf.reduce_sum(tf.square(mpr-tf.expand_dims(tm, 1)), 1) * n # B x 1

    # Between rater mean sqrs
    RMS = RSS / (k - 1) # B x 1

    # Between target sum sqrs
    BSS = tf.reduce_sum(tf.square(mpt-tf.expand_dims(tm, 1)), 1) * k # B x 1

    # Between target mean squares
    BMS = BSS / (n - 1) # B x 1

    # Residual sum of squares
    ESS = WSS - RSS # B x 1

    # Residual mean squares
    EMS = ESS / (n - 1) / (k - 1)

    if case == 3:
        if typ == 1:
            res = (BMS - EMS) / (BMS + (k - 1) * EMS)

    res # B x 1

    return res
