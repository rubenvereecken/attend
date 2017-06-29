import tensorflow as tf
import numpy as np

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
    def __init__(self, filenames, batch_size, time_steps=20, feat_name='conflict',
            num_epochs=None, seq_q=True):
        self.filenames = filenames
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.feat_name = feat_name
        self.num_epochs = num_epochs
        self.seq_q = seq_q
        self.dim_feature = [224, 224, 3]

        self.scope = 'input'


    def input_producer(self):
        with tf.name_scope(self.scope):
            filename_q = tf.train.string_input_producer(
                    self.filenames, num_epochs=self.num_epochs, shuffle=True)

            example, target, context = self.read_and_decode_from_tfrecords(filename_q)
            return example, target, context


    def batch_sequences_with_states(self):
        example, target, context = self.input_producer()
        # TODO batch_sequences_with_states needs a shape, try to get rid of that?
        example.set_shape([None, np.prod(self.dim_feature)])

        with tf.name_scope(self.scope):
            # TODO this should really be like _initial_lstm in model.py
            # dim_conv = 14 * 14 * 512
            initial_states = {
                    'lstm_c': tf.zeros([512], dtype=tf.float32),
                    'lstm_h': tf.zeros([512], dtype=tf.float32),
                }

            batch = tf.contrib.training.batch_sequences_with_states(
                    input_sequences={
                        'images': example,
                        self.feat_name: target,
                    },
                    input_key      = context['subject'],
                    input_context  = context,
                    input_length   = tf.cast(context['num_frames'], tf.int32),
                    initial_states = initial_states,
                    num_unroll     = self.time_steps,
                    batch_size     = self.batch_size,
                    num_threads    = 2,
                    capacity       = self.batch_size * 2 * 2
                    )
            example_batch, target_batch = batch.sequences['images'], batch.sequences[self.feat_name]

            self.features    = example_batch
            self.targets     = target_batch
            self.state_saver = batch


    def read_and_decode_from_tfrecords(self, filename_q):
        feat_name = self.feat_name
        with tf.name_scope(self.scope):
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_q)
            # https://github.com/tensorflow/tensorflow/issues/976
            # http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
            context, feature_lists = tf.parse_single_sequence_example(
                    serialized_example,
                    context_features=dict(
                        subject    = tf.FixedLenFeature([], dtype = tf.string),
                        video      = tf.FixedLenFeature([], dtype = tf.string),
                        num_frames = tf.FixedLenFeature([], dtype = tf.int64)
                    ),
                    sequence_features={
                        'images'    : tf.FixedLenSequenceFeature([], dtype  = tf.string),
                        feat_name : tf.FixedLenSequenceFeature([1], dtype = tf.float32)
                    })

            images = tf.decode_raw(feature_lists['images'], tf.float32)
            context['subject'] = tf.Print(context['subject'], [context['subject']], message='video ')
            # images = tf.Print(images, [tf.shape(images)], message='images shape ')

            return images, feature_lists[feat_name], context
