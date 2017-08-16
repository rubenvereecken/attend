import tensorflow as tf

identity = lambda x, **kwargs: x

activations = dict(
    none=identity,
    tanh=tf.nn.tanh,
    sigmoid=tf.nn.sigmoid,
    relu=tf.nn.relu
)

def get_activation(s):
    if s not in activations:
        raise ValueError('Unknown activation `{}`'.format(s))
    return activations[s]
