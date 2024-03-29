import threading
import tensorflow as tf
import numpy as np

from collections import OrderedDict

from attend.log import Log; log = Log.get_logger(__name__)
import attend

# https://gist.github.com/jimfleming/d1118cc630f5c883223a4b4645cc2e7b
class GeneratorRunner:
    "Custom runner that that runs an generator in a thread and enqueues the outputs."

    def __init__(self, generator, placeholders, enqueue_op, close_op, name=None):
        self._generator = generator
        self._placeholders = placeholders
        self._enqueue_op = enqueue_op
        self._close_op = close_op
        self.name = name if name else 'provider'

    def _run(self, sess, coord):
        try:
            while not coord.should_stop():
                try:
                    values = next(self._generator)

                    assert len(values) == len(self._placeholders), \
                        'generator values and placeholders must have the same length'

                    feed_dict = {placeholder: value \
                        for placeholder, value in zip(self._placeholders, values)}
                    print('Running enqueue')
                    sess.run(self._enqueue_op, feed_dict=feed_dict)
                except (StopIteration, tf.errors.OutOfRangeError):
                    try:
                        print('Closing')
                        # sess.run(self._close_op)
                        print('Closed queue runner')
                    except Exception:
                        pass
                    return
        except Exception as ex:
            print('Something occurred in the generator runner')
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
        capacity=2, # Used to be 10
        num_epochs=None,
        shuffle_capacity=0,
        **kwargs):
    """
    Generates a single sequence from a sequence generator
    """

    with tf.variable_scope(scope):
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
        tf.train.add_queue_runner(queue_runner,
                                  collection=attend.GraphKeys.INPUT_RUNNERS)

        if summary_name:
            tf.summary.scalar(summary_name,
                    tf.cast(q.size(), tf.float32) * (1. / capacity))

        sample = q.dequeue()

        # Dequeueing loses shape, so set it back
        for i, tensor in enumerate(sample):
            tensor.set_shape(shapes[i])

        example, target, context = sample[0], sample[1], dict(key=sample[2], num_frames=sample[3])

        if shuffle_capacity > 0:
            example, target, context = \
                shuffle_data_and_context(example, target, context, shuffle_capacity, **kwargs)

        return example, target, context


def generate_single_sequence_example_from_hdf5(filename, target_name, scope, **kwargs):
    reader = HDF5SequenceReader(filename, target_name)
    return generate_single_sequence_example(reader, scope, **kwargs)


class HDF5SequenceReader:
    def __init__(self, filename, target_name):
        import h5py
        # TODO close
        f = h5py.File(filename, 'r')
        self.f = f
        self.features = f['features']
        self.targets = f[target_name]
        self.keys = list(self.features.keys())

        # TODO should communicate these shapes somewhere because they are fixed
        # Sequence, so ignore first dimension (time)
        self.feature_shape = self.features[self.keys[0]].shape[1:]
        self.target_shape = self.targets[self.keys[0]].shape[1:]
        self._generator = self._generator_fun()

    def close(self):
        self.f.close()

    def _generator_fun(self):
        for key in self.keys:
            # context = dict(key=key, num_examples=self.features[key].shape[0])
            # context = dict(key=key)
            yield self.features[key], self.targets[key], key, self.features[key].shape[0]

    def __iter__(self): return self

    def __next__(self):
        return next(self._generator)


def read_and_decode_from_tfrecords(filename_q, sequence_dims, scope):
    with tf.variable_scope('read_tfrecords'):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_q)
        # http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
        context, feature_lists = tf.parse_single_sequence_example(
                serialized_example,
                context_features={
                    'key'       : tf.FixedLenFeature([], dtype = tf.string),
                    'num_frames': tf.FixedLenFeature([], dtype=tf.int64)
                },
                sequence_features={
                    k: tf.FixedLenSequenceFeature([], dtype=tf.float32) \
                    for k, dims in sequence_dims.items()
                },
                # sequence_features={
                #     # 'features' : tf.FixedLenSequenceFeature([], dtype  = tf.string),
                #     'features': tf.FixedLenSequenceFeature([], dtype=tf.float32),
                #     target_name  : tf.FixedLenSequenceFeature([1], dtype = tf.float32)
                # },
                name='parse_sequence')

        # images = tf.decode_raw(feature_lists['features'], tf.float32)
        # images = feature_lists['features']
        if Log.debug:
            context['key'] = tf.Print(context['key'], [context['key'], context['num_frames']], message='video ')
        context['num_frames'] = tf.cast(context['num_frames'], tf.int32)

        return feature_lists, context


def shuffle_data_and_context(sequences, context, capacity, min_after_dequeue=None):
    q_in = sequences.copy()
    q_in.update(context)

    out = shuffle_queue(q_in, capacity, scope='shuffle_examples')
    sequences = { k: out.pop(k) for k in list(sequences.keys()) }
    context = out

    return sequences, context


def read_single_sequence_example_fom_tfrecord(filenames, sequence_dims, scope,
        shuffle_capacity=0, num_epochs=None, **kwargs):

    # with tf.name_scope('read_single_tfrecord'):
    filename_q = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs,
            name=kwargs.pop('name', 'filename_queue'))

    sequences, context = read_and_decode_from_tfrecords(filename_q, sequence_dims, scope)

    if shuffle_capacity > 0:
        sequences, context = \
            shuffle_data_and_context(sequences, context, shuffle_capacity, **kwargs)

    return sequences, context


def read_metadata_from_tfrecords(filename):
    """
    Reads the first sequence example's context to get features shape
    Assume it has a context value 'features.shape' that is an int list
    """
    import time
    start = time.time()
    # from google.protobuf.json_format import MessageToJson
    # MessageToXX encodes bytes as base64, so decode to string first
    from google.protobuf.json_format import MessageToDict, base64
    import simplejson as json

    raw_example = next(tf.python_io.tf_record_iterator(filename))
    # Don't read a sequence example, this will just read the context
    pb_context = tf.train.Example.FromString(raw_example)
    # context = json.loads(MessageToJson(pb_context))
    context = MessageToDict(pb_context)
    context = context['features']['feature']

    def _decode_bytes(v):
        return base64.b64decode(v).decode()

    try:
        feature_keys = list(map(_decode_bytes,
                                context['feature_keys']['bytesList']['value']))
    except:
        log.warning('No feature keys found in tfrecords, falling back to `features`')
        feature_keys = ['features']

    raw_shapes = OrderedDict((key, context['{}.shape'.format(key)]['int64List']['value']) \
                  for key in feature_keys)
    shapes = { key: tuple(map(int, raw_shape)) for key, raw_shape in raw_shapes.items() }
    print('Took {:.2f}s to infer feature shapes'.format(time.time()-start))
    return shapes


def shuffle_queue(d, capacity, min_after_dequeue=None, scope=None):
    """
    d should be an OrderedDict of tensors
    """
    if min_after_dequeue is None:
        min_after_dequeue = np.ceil(capacity / 2)

    with tf.name_scope(scope or 'shuffle'):
        dtypes = [v.dtype for v in d.values()]

        q = tf.RandomShuffleQueue(capacity=capacity,
                min_after_dequeue=min_after_dequeue,
                dtypes=dtypes,
                names=list(d.keys())
                )

        out = q.dequeue()

        shapes = { k: v.shape for k, v in d.items() }
        for k, v in out.items():
            v.set_shape(shapes[k])

        qr = tf.train.QueueRunner(q, enqueue_ops=[q.enqueue(d)],
                close_op=q.close(cancel_pending_enqueues=True))
        tf.train.add_queue_runner(qr)

        return out
