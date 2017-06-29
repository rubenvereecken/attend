import tensorflow as tf

def read_and_decode(filename_q, feat_name = 'conflict'):
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
    # context['subject'] = tf.Print(context['subject'], [context['subject']], message='video ')
    images = tf.Print(images, [tf.shape(images)], message='images shape ')

    return images, feature_lists[feat_name], context


def input_pipeline(filenames, batch_size, time_steps=20, feat_name='conflict', num_epochs=None, seq_q=True):
    # TODO is num_epochs parallel stuff?
    with tf.name_scope('input'):
        filename_q = tf.train.string_input_producer(
                filenames, num_epochs=num_epochs, shuffle=True)

        example, target, context = read_and_decode(filename_q, feat_name)

        # TODO this should really be like _initial_lstm in model.py
        # dim_conv = 14 * 14 * 512
        initial_states = {'lstm_states': tf.zeros([512], dtype=tf.float32)}
        # TODO batch_sequences_with_states needs a shape, try to get rid of that?
        example.set_shape([None, 224*224*3])

        if seq_q:
            batch = tf.contrib.training.batch_sequences_with_states(
                    input_key=context['subject'],
                    input_sequences={
                        'images': example,
                        feat_name: target,
                        },
                    input_context=context,
                    input_length=tf.cast(context['num_frames'], tf.int32),
                    initial_states=initial_states,
                    num_unroll=time_steps,
                    batch_size=batch_size,
                    num_threads=2,
                    capacity=batch_size * 2 * 2
                    )
            example_batch, target_batch = batch.sequences['images'], batch.sequences[feat_name]

        else:
            example_batch, target_batch = tf.train.batch(
                [example, target], batch_size=batch_size,
                num_threads=2,
                dynamic_pad=True,
                capacity=32 # TODO look into these values
                )

    return example_batch, target_batch
