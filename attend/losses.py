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

