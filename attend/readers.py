import threading
import tensorflow as tf

# https://gist.github.com/jimfleming/d1118cc630f5c883223a4b4645cc2e7b
class GeneratorRunner:
    "Custom runner that that runs an generator in a thread and enqueues the outputs."

    def __init__(self, generator, placeholders, enqueue_op, close_op):
        self._generator = generator
        self._placeholders = placeholders
        self._enqueue_op = enqueue_op
        self._close_op = close_op

    def _run(self, sess, coord):
        try:
            while not coord.should_stop():
                try:
                    values = next(self._generator)

                    assert len(values) == len(self._placeholders), \
                        'generator values and placeholders must have the same length'

                    feed_dict = {placeholder: value \
                        for placeholder, value in zip(self._placeholders, values)}
                    sess.run(self._enqueue_op, feed_dict=feed_dict)
                except (StopIteration, tf.errors.OutOfRangeError):
                    try:
                        sess.run(self._close_op)
                    except Exception:
                        pass
                    return
        except Exception as ex:
            if coord:
                coord.request_stop(ex)
            else:
                raise

    def create_threads(self, sess, coord=None, daemon=False, start=False):
        "Called by `start_queue_runners`."

        thread = threading.Thread(
            target=self._run,
            args=(sess, coord))

        if coord:
            coord.register_thread(thread)

        if daemon:
            thread.daemon = True

        if start:
            thread.start()

        return [thread]

def generate_single_sequence_example(
        reader,
        scope,
        summary_name='input_q',
        capacity=10):
    """
    Generates a single sequence from a sequence generator
    """

    with tf.name_scope(scope):
        dtypes = [tf.float32, tf.float32, tf.string, tf.int32]
        # features and targets are sequences of unknown length
        shapes = [tf.TensorShape([None]).concatenate(reader.feature_shape),
                tf.TensorShape([None]).concatenate(reader.target_shape),
                tf.TensorShape([]), tf.TensorShape([])]
        # TODO capacity
        q = tf.FIFOQueue(
                capacity=capacity,
                dtypes=dtypes)

        # TODO add names
        placeholders = [tf.placeholder(dtype, shape) for dtype, shape in zip(dtypes, shapes)]

        enqueue_op = q.enqueue(placeholders)
        close_op = q.close(cancel_pending_enqueues=True)
        queue_runner = GeneratorRunner(reader, placeholders, enqueue_op, close_op)

        # Keep in a separate collection so it can be started manually
        # ... because the supervisor keeps tripping up
        tf.train.add_queue_runner(queue_runner, collection='input_runners')

        if summary_name:
            tf.summary.scalar(summary_name,
                    tf.cast(q.size(), tf.float32) * (1. / capacity))

        sample = q.dequeue()

        # Dequeueing loses shape, so set it back
        for i, tensor in enumerate(sample):
            tensor.set_shape(shapes[i])

        # sample[2] = tf.Print(sample[2],[sample[2], sample[3]], message='sample')

        return sample[0], sample[1], dict(key=sample[2], num_frames=sample[3])


def generate_single_sequence_example_from_hdf5(filename, feat_name, scope, **kwargs):
    reader = HDF5SequenceReader(filename, feat_name)
    return generate_single_sequence_example(reader, scope, **kwargs)


class HDF5SequenceReader:
    def __init__(self, filename, feat_name):
        import h5py
        # TODO close
        f = h5py.File(filename, 'r')
        self.f = f
        self.features = f['features']
        self.targets = f[feat_name]
        self.keys = list(self.features.keys())

        # TODO should communicate these shapes somewhere because they are fixed
        # Sequence, so ignore first dimension (time)
        self.feature_shape = self.features[self.keys[0]].shape[1:]
        self.target_shape = self.targets[self.keys[0]].shape[1:]
        self._generator = self._generator_fun()

    def _generator_fun(self):
        for key in self.keys:
            # context = dict(key=key, num_examples=self.features[key].shape[0])
            # context = dict(key=key)
            yield self.features[key], self.targets[key], key, self.features[key].shape[0]

    def __iter__(self): return self

    def __next__(self):
        return next(self._generator)


def read_and_decode_from_tfrecords(filename_q, feat_name, scope):
    with tf.name_scope(scope):
        # TODO so why does this record reader not accept a capacity?
        # It's a queue right? So waddup
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
        context['key'] = context['subject']
        context['key'] = tf.Print(context['key'], [context['key'], context['num_frames']], message='video ')
        # images = tf.Print(images, [tf.shape(images)], message='images shape ')

        return images, feature_lists[feat_name], context


def read_single_sequence_example_fom_tfrecord(filename, feat_name, scope, **kwargs):
    with tf.name_scope(scope):
        filename_q = tf.train.string_input_producer(
                [filename], num_epochs=kwargs.get('num_epochs', None), shuffle=True)

        example, target, context = read_and_decode_from_tfrecords(filename_q, feat_name, scope)
        # print((example.shape, target.shape, context['subject']))
        return example, target, context