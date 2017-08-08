import tensorflow as tf

# The only pretty way to get access to a Dense class
import tensorflow.contrib.keras as K


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
    def __init__(self, num_units):
        self._num_units = num_units
        # W_a * s_i
        self._query_layer = K.layers.Dense(num_units, name='query_layer', use_bias=False)
        # U_a * h_j
        self._memory_layer = K.layers.Dense(num_units, name='memory_layer', use_bias=False)
        self._probability_fn = tf.nn.softmax


    def _align(self, memory, query):
        keys = self._memory_layer(memory)
        processed_query = self._query_layer(query)
        processed_query = tf.expand_dims(processed_query, 1) # B x 1 x D_attention
        v = tf.get_variable('attention_v', [self._num_units], tf.float32)

        # TODO normalize option

        # e_i
        score = v * tf.tanh(keys + processed_query) # B x T x D_attention

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
