import tensorflow as tf
import numpy as np

from attend.readers import *
from attend.log import Log; log = Log.get_logger(__name__)
from attend import util

class Provider():
    ENCODE_LSTM_C = 'encode_lstm_c'
    ENCODE_LSTM_H = 'encode_lstm_h'

    def __init__(self, filenames, encoder, batch_size, num_hidden, time_steps,
            feat_name='conflict',
            # num_epochs=None,
            seq_q=True, debug=False):
        self.filenames   = filenames
        self.batch_size  = batch_size
        self.time_steps  = time_steps
        self.T           = time_steps
        self.H           = num_hidden
        self.feat_name   = feat_name
        # self.num_epochs  = num_epochs
        self.seq_q       = seq_q
        self.dim_feature = [224, 224, 3]
        self.encoder     = encoder
        # Encoded dimension after flattening
        # self.scope       = 'input'
        # with tf.name_scope('input') as name_scope:
        #     self.name_scope = name_scope
        # with tf.variable_scope('input') as scope:
        #     self.scope = scope
        self.state_saver = None
        self.debug = debug

        if len(filenames) == 1 and filenames[0].endswith('hdf5'):
            # self.input_producer = generate_single_sequence_example_from_hdf5
            reader = HDF5SequenceReader(filenames[0], feat_name)
            self.dim_feature = reader.feature_shape
            # TODO not using scope atm
            self.input_producer = lambda filename, feat_name, scope, **kwargs: \
                generate_single_sequence_example(reader, scope, **kwargs)

        elif len(filenames) == 1 and filenames[0].endswith('tfrecords'):
            seq_shape = read_shape_from_tfrecords_for(filenames[0])
            # TODO just flatten it for now, might want shape back later
            self.dim_feature = (np.prod(seq_shape[1:]),)
            log.warning('%s', self.dim_feature)
            self.input_producer = read_single_sequence_example_fom_tfrecord
        else:
            print(filenames)
            raise Exception('Unknown file format, expecting just one file')

        self.dim_encoded = self._deduce_encoded_dim()


    # TODO seriously find a way to only build the convnet once this is ridiculous
    def _deduce_encoded_dim(self):
        dims = self.encoder.encoded_dim(self.dim_feature)
        if type(dims) == list: dims = np.prod(dims)
        return dims


    def batch_static_pad(self):
        example, target, context = self.input_producer(self.filenames[0], self.feat_name, self.name_scope)

        # TODO cheating
        self.dim_feature = (np.prod(self.dim_feature),)
        if len(self.dim_feature) + 1 != example.shape.ndims:
            example = tf.reshape(example, [-1, *self.dim_feature])
        example.shape.merge_with([None, *self.dim_feature])

        log.debug('example shape %s', example.shape)
        log.debug('target shape  %s', target.shape)

        # padding = tf.constant(self.T) - tf.shape(example)[0]
        padding = [[0,0],[0,0]]
        padding[1][1] = tf.constant(self.T) - tf.shape(example)[0]
        example = tf.pad(example, padding, 'CONSTANT')

        example_batch, target_batch = tf.train.batch(
            [example, target], batch_size=self.batch_size,
            num_threads=1, # Change if actually using this
            dynamic_pad=True,
            capacity=8 # TODO look into these values
            )

        self.features    = example_batch
        self.targets     = target_batch

        return example_batch, target_batch


    # TODO change everything to Data API as soon as more support comes in
    def batch_sequences_with_states(self, num_epochs=None, collection=None,
            container_name=None):
        if container_name:
            container = tf.container(container_name)
        else:
            container = util.noop()
        with container:
            with tf.variable_scope('input') as scope:
                g = tf.get_default_graph()
                example, target, context = self.input_producer(self.filenames[0],
                        self.feat_name, scope, num_epochs=num_epochs)


                # If mismatch, it probably needs a reshape
                with tf.name_scope('reshape'):
                    if len(self.dim_feature) + 1 != example.shape.ndims:
                        example = tf.reshape(example, [-1, *self.dim_feature])
                    example.shape.merge_with([None, *self.dim_feature])

                # tf.assert_equal(tf.shape(example)[0], tf.shape(target)[0])

                # NOTE it's important every state below is saved, otherwise it blocks
                with tf.variable_scope('initial'):
                    initial_states = {
                            'lstm_c': tf.zeros([self.H], dtype=tf.float32),
                            'lstm_h': tf.zeros([self.H], dtype=tf.float32),
                            # Keep the previous batch around too for extra history
                            'history': tf.zeros([self.T, np.prod(self.dim_encoded)], dtype=tf.float32),
                            'first': tf.constant(True)
                        }

                    if self.encoder.encode_lstm:
                        log.debug('Preparing encoder LSTM saved state')
                        initial_states.update({
                            Provider.ENCODE_LSTM_C: \
                                    tf.zeros([self.encoder.encode_hidden_units], dtype=tf.float32),
                            Provider.ENCODE_LSTM_H: \
                                    tf.zeros([self.encoder.encode_hidden_units], dtype=tf.float32),
                            })

                batch = tf.contrib.training.batch_sequences_with_states(
                        input_sequences={
                            'images': example,
                            self.feat_name: target,
                        },
                        input_key      = context['key'],
                        input_context  = context,
                        input_length   = context['num_frames'],
                        initial_states = initial_states,
                        num_unroll     = self.time_steps,
                        batch_size     = self.batch_size,
                        num_threads    = 2, # TODO change
                        capacity       = self.batch_size * 1,
                        name           = 'batch_seq_with_states'
                        )
                example_batch, target_batch = batch.sequences['images'], batch.sequences[self.feat_name]

                # Move the queue runners to a different collection
                if not collection is None:
                    runners = g.get_collection_ref(tf.GraphKeys.QUEUE_RUNNERS)
                    removed = [runners.pop(), runners.pop()]
                    for r in removed:
                        tf.train.add_queue_runner(r, collection)

                self.features    = example_batch
                self.targets     = target_batch
                self.state_saver = batch
                # This fixes an expectation of targets being single-dimensional further down the line
                # So like [?, T, 1] instead of just [?, T]
                if len(self.targets.shape) <= 2:
                    self.targets = tf.expand_dims(self.targets, -1)

