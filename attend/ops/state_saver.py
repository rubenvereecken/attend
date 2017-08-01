import tensorflow as tf
import numpy as np


from attend import ops


class ManualStateSaver:
    """
    Mimics the NextSequencedBatch API from Tensorflow
    """
    def __init__(self, input_sequences, input_key, input_context,
            input_length, initial_states, num_unroll, batch_size):

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
        self._length = tf.placeholder(tf.int64, (None,))
        self._sequence = tf.placeholder(tf.int32, (None,))
        self._sequence_count = tf.placeholder(tf.int32, (None,))

        # Complicated key, also to be fed in currently
        self._key = tf.placeholder(tf.string, (None,))


    def state(self, k):
        t = self._states[k].lookup(self._key)
        shape = self._state_shapes[k]
        return tf.reshape(t, [-1, *shape.as_list()])
        # return t

    def save_state(self, key, value, name=None):
        shape = self._state_shapes[key]
        value = tf.reshape(value, [-1, np.prod(shape.as_list()).astype(int)])
        value = tf.Print(value, [self._key], message='Saving {} '.format(key))
        return self._states[key].insert(self._key, value)


    def reset_states(self):
        deps = []
        n = tf.shape(self._key)[0]
        for k, v in self._initial_states.items():
            v = ops.repeat(v, n)
            deps.append(self.save_state(k, v))
        with tf.control_dependencies(deps):
            return tf.identity(self._key)

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
        return self._key

