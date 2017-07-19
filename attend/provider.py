import tensorflow as tf
import numpy as np

from attend.readers import *
from attend.log import Log; log = Log.get_logger(__name__)

# def input_pipeline():
#     with tf.name_scope('input'):

#         if seq_q:

#         else:
#             example_batch, target_batch = tf.train.batch(
#                 [example, target], batch_size=batch_size,
#                 num_threads=2,
#                 dynamic_pad=True,
#                 capacity=32 # TODO look into these values
#                 )

#     return example_batch, target_batch


class Provider():
    ENCODE_LSTM_C = 'encode_lstm_c'
    ENCODE_LSTM_H = 'encode_lstm_h'

    def __init__(self, filenames, encoder, batch_size, num_hidden, time_steps, feat_name='conflict',
            num_epochs=None, seq_q=True, debug=False):
        self.filenames   = filenames
        self.batch_size  = batch_size
        self.time_steps  = time_steps
        self.T           = time_steps
        self.H           = num_hidden
        self.feat_name   = feat_name
        self.num_epochs  = num_epochs
        self.seq_q       = seq_q
        self.dim_feature = [224, 224, 3]
        self.encoder     = encoder
        # Encoded dimension after flattening
        self.scope       = 'input'
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

        self.encoded_dim = self._deduce_encoded_dim()


    # TODO seriously find a way to only build the convnet once this is ridiculous
    def _deduce_encoded_dim(self):
        dims = self.encoder.encoded_dim(self.dim_feature)
        if type(dims) == list: dims = np.prod(dims)
        return dims


    def batch_static_pad(self):
        example, target, context = self.input_producer(self.filenames[0], self.feat_name, self.scope)

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


    def batch_sequences_with_states(self):
        example, target, context = self.input_producer(self.filenames[0], self.feat_name, self.scope)
        # TODO batch_sequences_with_states needs a shape, try to get rid of that?
        # example.set_shape([None, np.prod(self.dim_feature)])

        # If mismatch, it probably needs a reshape
        if len(self.dim_feature) + 1 != example.shape.ndims:
            example = tf.reshape(example, [-1, *self.dim_feature])
        example.shape.merge_with([None, *self.dim_feature])

        # TODO
        # This is just for debugging when source and target sequence don't match in len
        # if self.debug:
        #     min_time_steps = tf.minimum(tf.shape(example)[0], tf.shape(target)[0])
        #     example = example[:min_time_steps]
        #     target = target[:min_time_steps]

        with tf.name_scope(self.scope):
            # TODO this should really be like _initial_lstm in model.py
            # dim_conv = 14 * 14 * 512

            # NOTE it's important every state below is saved, otherwise it blocks
            initial_states = {
                    'lstm_c': tf.zeros([self.H], dtype=tf.float32),
                    'lstm_h': tf.zeros([self.H], dtype=tf.float32),
                    # Keep the previous batch around too for extra history
                   # 'history': tf.zeros([self.batch_size, self.T, np.prod(self.encoded_dim)], dtype=tf.float32),
                    # 'first': tf.constant(True)
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
                    input_length   = tf.cast(context['num_frames'], tf.int32),
                    initial_states = initial_states,
                    num_unroll     = self.time_steps,
                    batch_size     = self.batch_size,
                    num_threads    = 1, # TODO change
                    capacity       = self.batch_size
                    )
            example_batch, target_batch = batch.sequences['images'], batch.sequences[self.feat_name]

            self.features    = example_batch
            self.targets     = target_batch
            self.state_saver = batch
            # This fixes an expectation of targets being single-dimensional further down the line
            # So like [?, T, 1] instead of just [?, T]
            if len(self.targets.shape) <= 2:
                self.targets = tf.expand_dims(self.targets, -1)
