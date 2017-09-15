import tensorflow as tf
import numpy as np

from functools import partial

import attend
from attend import readers, util, tf_util
from attend.log import Log
log = Log.get_logger(__name__)


class Provider():
    ENCODE_LSTM_C = 'encode_lstm_c'
    ENCODE_LSTM_H = 'encode_lstm_h'

    def __init__(self, encoder, batch_size, num_hidden, time_steps,
                 target_name='conflict',
                 sequence_dims=None,
                 shuffle_examples=False, shuffle_examples_capacity=None,
                 learn_initial_states=False,
                 num_image_patches=None,
                 debug=False):
        self.batch_size           = batch_size
        self.T                    = time_steps
        self.L                    = num_image_patches
        self.time_steps           = time_steps
        self.H                    = num_hidden
        self.num_hidden           = num_hidden
        self.target_name          = target_name
        self.sequence_dims        = sequence_dims
        self.encoder              = encoder
        self.debug                = debug
        self.learn_initial_states = learn_initial_states

        self.state_saver          = None

        self.shuffle_examples = shuffle_examples
        self.shuffle_examples_capacity = shuffle_examples_capacity \
            if shuffle_examples_capacity is not None else batch_size * 4

        self.dim_encoded = self._deduce_encoded_dim()
        self.dim_patch = { k: None if self.L is None else int(self.dim_encoded[k] / self.L) for k in self.feature_names }

        # Overridden in InMemoryProvider
        self._batch_sequences_with_states = tf.contrib.training.batch_sequences_with_states

    def input_producer(*args, **kwargs):
        raise NotImplementedError()

    def _deduce_encoded_dim(self):
        encoded_dims = {}
        for k, sequence_dims in self.feature_dims.items():
            dims = self.encoder.encoded_dim(sequence_dims)
            if isinstance(dims, list):
                dims = np.prod(dims)
            encoded_dims[k] = dims

        return encoded_dims

    def _prepare_initial(self, is_training, reuse=False, scope=None):
        with tf.variable_scope('init_constant'):
            initial_constants = {
                # Keep the previous batch around too for extra history
                'output': tf.zeros([1], dtype=tf.float32),
                'target': tf.zeros([1], dtype=tf.float32),
                'first': tf.constant(True)
            }

            initial_feat_varnames = ['lstm_c', 'lstm_h', 'context']

            for key in self.feature_names:
                initial_constants.update({
                    '{}.lstm_c'.format(key): tf.zeros([self.H], dtype=tf.float32),
                    '{}.lstm_h'.format(key): tf.zeros([self.H], dtype=tf.float32),
                    '{}.context'.format(key): tf.zeros([self.dim_patch[key] or \
                                                        self.dim_encoded[key]],
                                                       dtype=tf.float32),
                    # Not learned tho
                    '{}.history'.format(key): tf.zeros(
                        [self.T, self.dim_encoded[key]], dtype=tf.float32),
                        # [self.T, np.prod(list(self.dim_encoded[key]))], dtype=tf.float32),
                })

                if self.encoder.encode_lstm:
                    log.debug('Preparing encoder LSTM saved state')
                    initial_constants.update({
                        key + '.' + Provider.ENCODE_LSTM_C:
                        tf.zeros([self.encoder.encode_hidden_units], dtype=tf.float32),
                        key + '.' + Provider.ENCODE_LSTM_H:
                        tf.zeros([self.encoder.encode_hidden_units], dtype=tf.float32),
                    })
                    initial_feat_varnames.extend([Provider.ENCODE_LSTM_C, Provider.ENCODE_LSTM_H])

        initial_variables = {}
        def _create_variable(k, v):
            initial_variables[k] = tf.get_variable('initial_{}'.format(k),
                                                   # initializer=lambda *args, **kwargs: v, \
                                                   initializer=v,
                                                   trainable=is_training and self.learn_initial_states,
                                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                                attend.GraphKeys.INITIAL_STATES]) \

        with tf.variable_scope(scope or 'init_var', reuse=reuse):
            random_feature_name = self.feature_names[0]

            for key in ['output', 'target']:
                _create_variable(key, initial_constants[key])
            # This will only work as long as encoded are the same size
            for key in initial_feat_varnames:
                _create_variable(key, initial_constants[random_feature_name + '.' + key])

            if is_training:
                for k, v in initial_variables.items():
                    tf.summary.histogram(k, v.initialized_value())
                    # tf.summary.histogram(k, v)

        return initial_constants, initial_variables

    def preprocess_example(self, key, example):
        return example

    def batch_sequences_with_states(self, num_epochs=None, is_training=True, reuse=False, collection=None):
        container = util.noop()
        with container:
            with tf.variable_scope('input') as scope:
                g = tf.get_default_graph()

                shuffle_capacity = 0 if not self.shuffle_examples \
                    else self.shuffle_examples_capacity
                shuffle_min = max(self.batch_size, np.ceil(.75 * shuffle_capacity))
                if shuffle_capacity > 0:
                    log.info('Shuffling examples w capacity %s, min %s', shuffle_capacity, shuffle_min)

                sequences, context = self.input_producer(scope, num_epochs=num_epochs,
                                                         shuffle_capacity=shuffle_capacity,
                                                         min_after_dequeue=shuffle_min)

                for key in self.feature_names:
                    sequences[key] = self.preprocess_example(key, sequences[key])

                initial_states, initial_variables = self._prepare_initial(is_training, reuse)
                self.initial_variables = initial_variables

                batch = self._batch_sequences_with_states(
                    input_sequences   = sequences,
                    input_key         = context['key'] + ':',  # : for split
                    input_context     = context,
                    initial_states    = initial_states,
                    input_length      = context['num_frames'],
                    num_unroll        = self.time_steps,
                    batch_size        = self.batch_size,
                    num_threads       = 2,  # TODO change
                    capacity          = self.batch_size * 1,
                    name              = 'batch_seq_with_states',
                    make_keys_unique  = True,
                    allow_small_batch = True  # Required otherwise blocks
                )

                # Move the queue runners to a different collection
                if collection is not None:
                    runners = g.get_collection_ref(tf.GraphKeys.QUEUE_RUNNERS)
                    removed = [runners.pop(), runners.pop()]
                    for r in removed:
                        tf.train.add_queue_runner(r, collection)

                # self.features = { k: tf.identity(batch.sequences[k], name=k) \
                #                  for k in self.feature_names }
                self.features = { k: batch.sequences[k] for k in self.feature_names }
                self.targets = batch.sequences.get(self.target_name, None)
                self.state_saver = batch
                tf_util.add_to_collection(attend.GraphKeys.INPUT,
                                          list(self.features.values()))
                tf_util.add_to_collection(attend.GraphKeys.INPUT, self.targets)
                # This fixes an expectation of targets being single-dimensional
                # So like [?, T, 1] instead of just [?, T]
                if self.targets is not None and len(self.targets.shape) <= 2:
                    self.targets = tf.expand_dims(self.targets, -1)

                if self.learn_initial_states:
                    self._overwrite_initial(initial_variables)

    def _overwrite_initial(self, initial_variables):
        initial_variables = initial_variables.copy()
        first = self.state_saver.state('first')
        with tf.name_scope('prepare_overwriting'):
            only_first_mask = tf.expand_dims(first, 1)
            not_first_mask = tf.logical_not(only_first_mask)
            only_first_mask = tf.cast(only_first_mask, tf.float32, name='only_first_mask')
            not_first_mask = tf.cast(not_first_mask, tf.float32, name='not_first_mask')
            batch_size = tf.gather(tf.shape(first), 0, name='batch_size')

        vs = {}

        def _overwrite(k, init_v):
            v = self.state_saver.state(k)
            with tf.name_scope('assign_first_{}'.format(k)):
                # B x 1 * 1 x 512 -> B x 512 (repeats row)
                init_v = tf.einsum('ij,jk->ik', tf.ones([batch_size, 1]),
                                   tf.expand_dims(init_v, 0))
                init_v = only_first_mask * init_v
                v = not_first_mask * v
                v = v + init_v
                vs[k] = v

        for k in ['output', 'target']:
            _overwrite(k, initial_variables.pop(k))

        # There is no longer a 1-1 mapping between variables and initial states
        # For each feature, the learned variable e.g. 'lstm_c' should be the same
        for k, v in initial_variables.items():
            for name in self.feature_names:
                key = name + '.' + k
                _overwrite(key, v)

        self.states = vs

    def state(self, key):
        if self.learn_initial_states and key in self.states:
            # Holds a fixed version
            return self.states[key]
        else:
            return self.state_saver.state(key)

    def save_state(self, key, values):
        return self.state_saver.save_state(key, values)

    def batch_static_pad(self):
        example, target, context = self.input_producer(self.filenames[0], self.target_name, self.name_scope)

        # TODO cheating
        self.dim_feature = (np.prod(self.dim_feature),)
        if len(self.dim_feature) + 1 != example.shape.ndims:
            example = tf.reshape(example, [-1, *self.dim_feature])
        example.shape.merge_with([None, *self.dim_feature])

        log.debug('example shape %s', example.shape)
        log.debug('target shape  %s', target.shape)

        # padding = tf.constant(self.T) - tf.shape(example)[0]
        padding = [[0, 0], [0, 0]]
        padding[1][1] = tf.constant(self.T) - tf.shape(example)[0]
        example = tf.pad(example, padding, 'CONSTANT')

        example_batch, target_batch = tf.train.batch(
            [example, target], batch_size=self.batch_size,
            num_threads=1,  # Change if actually using this
            dynamic_pad=True,
            capacity=8  # TODO look into these values
        )

        self.features = example_batch
        self.targets = target_batch

        return example_batch, target_batch


class FileProvider(Provider):
    def __init__(self, filenames, *args, target_name='conflict', **kwargs):
        self.filenames = filenames
        # super().__init__(*args, **kwargs)

        if len(filenames) == 1 and filenames[0].endswith('hdf5'):
            # self.input_producer = generate_single_sequence_example_from_hdf5
            reader = readers.HDF5SequenceReader(filenames[0], target_name)
            dim_feature = reader.feature_shape
            # self.input_producer = lambda filename, target_name, scope, **kwargs: \
            #     generate_single_sequence_example(reader, scope, **kwargs)
            self.input_producer = partial(readers.generate_single_sequence_example,
                                          reader)

        elif all(map(lambda f: f.endswith('tfrecords'), filenames)):
            feature_shapes = readers.read_metadata_from_tfrecords(filenames[0])
            self.feature_names = list(feature_shapes.keys())
            # TODO just flatten it for now, might want shape back later
            self.feature_dims = { k: (np.prod(shape[1:]), ) for k, shape in feature_shapes.items() }
            sequence_dims = self.feature_dims.copy()
            sequence_dims.update({target_name: [1]}) # Prettier way?
            # log.warning('%s', dim_feature)
            self.input_producer = partial(readers.read_single_sequence_example_fom_tfrecord,
                                          filenames, sequence_dims)

        else:
            raise Exception('Unsupported file format')

        assert sequence_dims is not None
        kwargs['sequence_dims'] = sequence_dims
        kwargs['target_name'] = target_name
        super().__init__(*args, **kwargs)

    def preprocess_example(self, key, example):
        # If mismatch, it probably needs a reshape
        dim_feature = self.sequence_dims[key]
        with tf.name_scope('reshape'):
            if len(dim_feature) + 1 != example.shape.ndims:
                example = tf.reshape(example, [-1, *dim_feature])
            example.shape.merge_with([None, *dim_feature])
            return example


def batch_sequences_with_states(
        # These are all the arguments I'm definitely ignoring
        capacity, num_threads, make_keys_unique,
        allow_small_batch, name,
        **kwargs):
    from .ops.state_saver import ManualStateSaver
    state_saver = ManualStateSaver(**kwargs)
    return state_saver


class InMemoryProvider(Provider):
    def __init__(self, feature_dims={}, *args, **kwargs):
        self.feature_dims = feature_dims
        self.feature_names = list(feature_dims.keys())

        super().__init__(*args, **kwargs)

        # if 'sequence_dims' not in kwargs:
        #     raise Exception('MemoryProvider requires feature dimensions')

        # Gets populated by the input producer
        self.sequence_placeholders = {}
        self.context_placeholders = {}

        self.input_producer = self._placeholder_provider
        self._batch_sequences_with_states = batch_sequences_with_states

    def _placeholder_provider(self, scope, **kwargs):
        # Assume test for now
        with tf.variable_scope(scope):
            self.context_placeholders.update({
                'key': tf.placeholder(tf.string, shape=(None,),
                                      name='key'),
                'num_frames': tf.placeholder(tf.int64, shape=(None,),
                                             name='num_frames'),
            })

            # TODO dims is actually a singleton tuple because I flattened it earlier
            # Maybe one day I have a use for more dimensions again
            for k, dims in self.feature_dims.items():
                self.sequence_placeholders[k] = tf.placeholder(tf.float32, shape=(None, None, *dims),
                                                      name=k)

            self.sequence_placeholders[self.target_name] = tf.placeholder(tf.float32,
                                                          shape=(None, None,), name=self.target_name)

            return self.sequence_placeholders, self.context_placeholders

    def preprocess_example(self, key, example):
        return example
