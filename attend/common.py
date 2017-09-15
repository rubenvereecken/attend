import tensorflow as tf


def identity(x, **kwargs):
    return x


activations = dict(
    none=identity,
    tanh=tf.nn.tanh,
    sigmoid=tf.nn.sigmoid,
    relu=tf.nn.relu
)

import attend.losses

loss_functions = dict(
    mse=tf.losses.mean_squared_error,
    rmse=attend.losses.root_mean_squared_error,
    icc=attend.losses.icc_loss(3,1),
    # temp thing
    icc_unweighted=attend.losses.icc_loss(3,1,False)
)

def get_activation(s):
    if s not in activations:
        raise ValueError('Unknown activation `{}`'.format(s))
    return activations[s]


def get_loss_function(s):
    if s not in loss_functions:
        raise ValueError('Unknown loss function `{}`'.format(s))
    return loss_functions[s]
