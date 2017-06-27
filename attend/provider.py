import tensorflow as tf

def read_and_decode(filename_q, feat_name = 'conflict'):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_q)
    # https://github.com/tensorflow/tensorflow/issues/976
    # http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
    _, feature_lists = tf.parse_single_sequence_example(
            serialized_example,
            sequence_features={
                'images': tf.FixedLenSequenceFeature([], dtype=tf.string),
                feat_name: tf.FixedLenSequenceFeature([1], dtype=tf.float32)
                }
            )

    feature_lists['images'] = tf.decode_raw(feature_lists['images'], tf.float32)

    return feature_lists['images'], feature_lists[feat_name]
    # return None, feature_lists[feat_name]


def input_pipeline(filenames, batch_size, num_epochs=None):
    # TODO is num_epochs parallel stuff?
    with tf.name_scope('input'):
        filename_q = tf.train.string_input_producer(
                filenames, num_epochs=num_epochs, shuffle=True)

        example, target = read_and_decode(filename_q)

        # TODO pad
        # example_batch, target_batch = tf.train.batch(
        example_batch, target_batch = tf.train.batch(
            [example, target], batch_size=batch_size,
            num_threads=2,
            dynamic_pad=True,
            capacity=32 # TODO look into these values
            )
            # min_after_dequeue=10000)

    return example_batch, target_batch
    # return None, target_batch
