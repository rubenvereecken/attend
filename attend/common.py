import tensorflow as tf
from functools import partial


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

def deep_lstm(num_units):
    return tf.contrib.rnn.MultiRNNCell([
        tf.contrib.rnn.LSTMCell(num_units, use_peepholes=True) for _ in range(2)
    ])

lstm_implementations = dict(
    # Doesn't have peepholes
    basic=tf.contrib.rnn.BasicLSTMCell,
    basic_lstm=tf.contrib.rnn.BasicLSTMCell,

    norm_basic=tf.contrib.rnn.LayerNormBasicLSTMCell,

    peepholes=partial(tf.contrib.rnn.LSTMCell, use_peepholes=True),
    peep_lstm=partial(tf.contrib.rnn.LSTMCell, use_peepholes=True),
    lstm=partial(tf.contrib.rnn.LSTMCell, use_peepholes=True),

    deep=deep_lstm,
    deep_lstm=deep_lstm
)

def get_lstm_implementation(s):
    if s not in lstm_implementations:
        raise ValueError('Unknown lstm implementation `{}`').format(s)
    return lstm_implementations[s]
