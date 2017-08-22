import tensorflow as tf
import numpy as np


import attend
from attend import ops, tf_util


class ManualStateSaver:
    """
    Mimics the NextSequencedBatch API from Tensorflow
    """
    def __init__(self, input_sequences, input_key, initial_states,
            input_context={}, # Not supported
            input_length=None, num_unroll=None, batch_size=None):

        self._initial_states = initial_states
        self._states = {}
        self._sequences = input_sequences
        self._state_shapes = {}

        for k, v in initial_states.items():
            self._state_shapes[k] = v.shape
            # Have to store values flat for some reason
            value = tf.reshape(v, [-1])
            self._states[k] = tf.contrib.lookup.MutableHashTable(
                    key_dtype=tf.string, value_dtype=v.dtype,
                    default_value=value, name='{}_table'.format(k))

        # Properties to be fed in
        self._length = tf.placeholder(tf.int64, (None,), name='length')
        self._sequence = tf.placeholder(tf.int32, (None,), name='sequence')
        self._sequence_count = tf.placeholder(tf.int32, (None,), name='sequence_count')

        # Complicated key, also to be fed in currently
        self._full_key = tf.placeholder(tf.string, (None,), name='full_key')
        self._key = tf.placeholder(tf.string, (None,), name='key')

        tf_util.add_to_collection(attend.GraphKeys.STATE_SAVER,
                [#*self._sequences.values(),
                 self._full_key, self._key,
                 self._length, self._sequence, self._sequence_count])


    def state(self, k):
        with tf.name_scope(k):
            t = self._states[k].lookup(self._key)
            shape = self._state_shapes[k]
            t = tf.reshape(t, [-1, *shape.as_list()])
            return t

    def save_state(self, key, value, name=None):
        with tf.name_scope(name or 'save_{}'.format(key)):
            shape = self._state_shapes[key]
            value = tf.reshape(value, [-1, np.prod(shape.as_list()).astype(int)])
            # TODO check the value passed in
            # value = tf.Print(value, [self._key], message='Saving {} '.format(key))
            # if key != 'first':
            #     value = tf.Print(value, [tf.reduce_mean(value)], message='mean {} '.format(key))
            insert = self._states[key].insert(self._key, value)
            return insert


    def reset_states(self):
        with tf.name_scope('reset'):
            deps = []
            n = tf.shape(self._key)[0]
            for k, v in self._initial_states.items():
                with tf.name_scope(k):
                    v = ops.repeat(v, n)
                    deps.append(self.save_state(k, v))

            return tf.group(*deps, name='reset_group')

    @property
    def context(self):
        raise NotImplementedError('Context not supported atm')

    @property
    def sequences(self):
        return self._sequences

    @property
    def length(self):
        return self._length

    @property
    def sequence(self):
        return self._sequence

    @property
    def sequence_count(self):
        return self._sequence_count

    @property
    def key(self):
        return self._full_key

