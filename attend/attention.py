import tensorflow as tf

# The only pretty way to get access to a Dense class
import tensorflow.contrib.keras as K

import inspect

from attend import Log; log = Log.get_logger(__name__)
from attend.util import params_for


class Attention:
    def __init__(self):
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)


    def __call__(self, x, h, reuse=False):
        """
        x: memory; encoded input
        h: query; attention rnn output
        """
        raise NotImplementedError()


class BahdanauAttention(Attention):
    def __init__(self, num_units, normalize=False):
        self._num_units = num_units
        self._normalize = normalize
        # W_a * s_i
        self._query_layer = K.layers.Dense(num_units, name='query_layer', use_bias=False)
        # U_a * h_j
        self._memory_layer = K.layers.Dense(num_units, name='memory_layer', use_bias=False)
        self._probability_fn = tf.nn.softmax

        super().__init__()

        l = locals()
        # frame = inspect.currentframe()
        classname = self.__class__.__name__
        params = params_for(BahdanauAttention)
        args = [l[param] for param in params]
        s = ', '.join(['{}={}'.format(k, v) for k, v in zip(params, args)])
        s = '{}({})'.format(classname, s)
        log.info(s)


    def _align(self, memory, query):
        keys = self._memory_layer(memory)
        processed_query = self._query_layer(query)
        processed_query = tf.expand_dims(processed_query, 1) # B x 1 x D_attention
        v = tf.get_variable('attention_v', [self._num_units], tf.float32)
        b = tf.get_variable('attention_b', [self._num_units], tf.float32)

        if self._normalize:
            g = tf.get_variable('attention_g',
                    initializer=tf.sqrt(1. / self._num_units))
            # normed_v = g * v / ||v||
            v = g * v / tf.norm(v)

        # e_i
        score = v * tf.tanh(keys + processed_query + b) # B x T x D_attention

        # NOTE if you don't summarize over the features,
        # it's like attention per time per feature. Boom
        reduced_score = tf.reduce_sum(score, axis=2) # Reduce along D_attention axis

        return reduced_score # B x T


    def __call__(self, memory, query, reuse=False):
        """
        memory: B x T x H_enc (h in Bahdanau paper)
        query:  B x H_dec     (s in Bahdanau paper)
        """
        with tf.variable_scope('bahdanau_attention', reuse=reuse):
            score = self._align(memory, query)
            alpha = self._probability_fn(score, name='alpha') # B x T
            # expanded_alpha = tf.expand_dims(alpha, 1) # B x 1 x T

            # c_i = sum(1..T) alpha_ij * h_j
            # context = tf.reduce_sum(expanded_alpha * memory, axis=1, name='context')
            context = tf.einsum('ij,ijl->il', alpha, memory)
            # context = tf.matmul(expanded_alpha, memory) # B x 1 x H_enc
            # context = tf.squeeze(context, 1) # B x H_enc

            return context, alpha
