import tensorflow as tf
import numpy as np

# The only pretty way to get access to a Dense class
import tensorflow.contrib.keras as K

# import inspect

from attend import Log
from attend.util import params_for
from attend.common import get_activation

log = Log.get_logger(__name__)


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
    def __init__(self, num_units, normalize=False, score_nonlinearity=None):
        super().__init__()
        self._num_units = num_units
        self._normalize = normalize
        # W_a * s_i
        self._query_layer = K.layers.Dense(num_units, name='query_layer',
                                           kernel_initializer=self.weight_initializer,
                                           use_bias=False)
        # U_a * h_j
        self._memory_layer = K.layers.Dense(num_units, name='memory_layer',
                                            kernel_initializer=self.weight_initializer,
                                            use_bias=False)
        self._probability_fn = tf.nn.softmax

        if score_nonlinearity is None:
            self._score_nonlinearity = tf.nn.tanh
        elif isinstance(score_nonlinearity, str):
            self._score_nonlinearity = get_activation(score_nonlinearity)

        loc = locals()
        # frame = inspect.currentframe()
        classname = self.__class__.__name__
        params = params_for(BahdanauAttention)
        args = [loc[param] for param in params]
        s = ', '.join(['{}={}'.format(k, v) for k, v in zip(params, args)])
        s = '{}({})'.format(classname, s)
        log.info(s)

    def _score(self, memory, query):
        with tf.name_scope('align'):
            keys = self._memory_layer(memory)
            processed_query = self._query_layer(query)
            processed_query = tf.expand_dims(
                processed_query, 1)  # B x 1 x D_attention
            v = tf.get_variable('attention_v', [self._num_units], tf.float32,
                                initializer=self.weight_initializer)
            b = tf.get_variable('attention_b', [self._num_units], tf.float32,
                                initializer=self.const_initializer)

            if self._normalize:
                g = tf.get_variable('attention_g',
                                    initializer=tf.sqrt(1. / self._num_units))
                # normed_v = g * v / ||v||
                v = g * v / tf.norm(v)

            # e_i
            # B x T x D_attention
            score = v * self._score_nonlinearity(keys + processed_query + b)
            # Reduce along D_att axis
            # Score computation equal to einsum('s,bts->bt', v, ...)
            reduced_score = tf.reduce_sum(
                score, axis=2)

            # NOTE if you don't summarize over the features,
            # it's like attention per time per feature. Boom

        return reduced_score  # B x T

    def __call__(self, memory, query, reuse=False):
        """
        memory: B x T x H_enc (h in Bahdanau paper)
        query:  B x H_dec     (s in Bahdanau paper)
        """
        with tf.variable_scope('bahdanau_attention', reuse=reuse):
            score = self._score(memory, query)  # B x T
            alpha = self._probability_fn(score, name='alpha')  # B x T
            # expanded_alpha = tf.expand_dims(alpha, 1) # B x 1 x T

            # c_i = sum(1..T) alpha_ij * h_j
            # context = tf.reduce_sum(expanded_alpha * memory, axis=1,
            #    name='context')
            context = tf.einsum('ij,ijl->il', alpha, memory)
            # context = tf.matmul(expanded_alpha, memory) # B x 1 x H_enc
            # context = tf.squeeze(context, 1) # B x H_enc

            return context, alpha


class OldTimeNaive(Attention):
    """
    A version without a separate attention dimensionality
    Indeed the attention dimensionality is simply the encoder output dim
    Differences:
        - originally used relu activation
        - originally projected features x onto their own dimension
            ( I should add that back )
        - Works on the D_enc dimension, not D_attention, which isn't used
        - Since there is no D_attention, the memory x does not get projected
    """

    def __call__(self, x, h, reuse=False):
        """
        h: decoder hidden state from previous step
        """
        # Require features to be flat at this point
        x.shape.assert_has_rank(3)
        D_enc = np.prod(x.shape.as_list()[2:])
        H = h.shape.as_list()[-1]

        with tf.variable_scope('attention_layer', reuse=reuse):
            W = tf.get_variable(
                'W', [H, D_enc],
                initializer=self.weight_initializer)
            b = tf.get_variable(
                'b', [D_enc],
                initializer=self.const_initializer)
            h_att = tf.nn.relu(
                x + tf.expand_dims(tf.matmul( h, W), 1) + b,
                name='h_att')

            w_att = tf.get_variable('w_att', [D_enc, 1],
                initializer=self.weight_initializer)
            out_att = tf.einsum('ijk,kl->ij', h_att, w_att)
            # Softmax assigns probability to each frame
            alpha = tf.nn.softmax(out_att, name='alpha')
            context = tf.reduce_sum(
                x * tf.expand_dims(alpha, 2),
                1, name='context')

            return context, alpha
