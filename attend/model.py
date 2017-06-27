import tensorflow as tf
import tensorflow.contrib.keras as K
from tensorflow.contrib.keras import applications as apps

from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream

from callbacks import TensorBoard
from schemes import InfiniteSequentialBatchIterator as InfSeqBatchIterator


class AttendModel():
    def __init__(self, batch_size=None):

        # self.batch_size = batch_size

        # self.features = tf.placeholder(tf.float32, [batch_size, *dim_feat])
        # self.targets = tf.placeholder(tf.float32, [batch_size, n_time_step])
        pass


    # TODO split this up in predict and loss or something
    def build_model(self, features, targets):
        """Build the entire model

        Args:
            features: Feature batch Tensor (from provider)
            targets: Targets batch Tensor
        """

        print(features, targets)
        batch_size = tf.shape(targets)[0]
        # batch_size = features

        # print(tf.random_normal([]))
        loss = tf.Variable(tf.random_normal([]))
        loss = tf.Print(loss, [batch_size], message='batch size')

        return loss


def linear_reg(nb_outputs, dropout=.5, norm=True, sizes=[512, 512]):

    # Fit the Keras way of doing thangs
    def _network(net):
        for size in sizes:
            net = K.layers.Dense(size)(net)
            net = K.layers.BatchNormalization(axis=-1)(net)
            net = K.layers.Dropout(dropout)(net)

        net = K.layers.Dense(nb_outputs, activation='softmax')(net)
        return net

    return _network


def _target_path_for(target):
    return '/{}/V/aligned'.format(target)


def setup_model(
        target,
        data_file,
        batch_size,
        steps_per_epoch,
        epochs,
        log_dir
        ):
    # Define the datasets
    # Select the features and a single target
    sources = ['img', _target_path_for(target)]
    tr_set = H5PYDataset(data_file, which_sets=('train',), sources=sources)
    te_set = H5PYDataset(data_file, which_sets=('test',), sources=sources)

    # I feel like this can be done better
    tr_scheme = InfSeqBatchIterator(examples=tr_set.num_examples,
            batch_size=batch_size)
    te_scheme = InfSeqBatchIterator(examples=te_set.num_examples,
            batch_size=batch_size)

    tr_stream = DataStream(dataset=tr_set, iteration_scheme=tr_scheme)
    te_stream = DataStream(dataset=te_set, iteration_scheme=te_scheme)

    # input_layer = K.layers.Input(tr_set.source_shapes[0].shape[1:])
    base_model = apps.ResNet50(weights = 'imagenet')
    input_layer = base_model.input
    last_layer = base_model.get_layer('flatten_1').output
    pred = linear_reg(1)(last_layer)

    model = K.models.Model([input_layer], pred)

    model.compile(
            optimizer = K.optimizers.Adadelta(
                lr = 1.,
                rho = .95,
                epsilon = 1e-8,
                decay = 5e-5,
                ),
            loss = K.losses.mean_squared_error
            )

    model.fit_generator(
            generator=tr_stream.get_epoch_iterator(),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            max_q_size=10,
            # nb_val_samples=100,
            validation_data=te_stream.get_epoch_iterator(),
            validation_steps=100,
            callbacks=[
                # K.callbacks.TensorBoard(log_dir=log_dir),
                TensorBoard(log_dir=log_dir),
                K.callbacks.CSVLogger(filename=log_dir + '/logger.csv'),
                K.callbacks.ModelCheckpoint(log_dir + '/model.h5'),
                ]
            )

    return model
