import tensorflow as tf
import tensorflow.contrib.keras as K
from tensorflow.contrib.keras import applications as apps

from fuel.datasets.hdf5 import H5PYDataset

# from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D, BatchNormalization, RepeatVector
# from keras.models import Sequential
# from ..layers import linREG, softmaxPDF, ordinalPDF

from schemes import InfiniteSequentialBatchIterator as InfSeqBatchIterator

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
            generator=DataStream(dataset=tr_set, iteration_scheme=tr_scheme,
            validation_data=DataStream(dataset=te_set, iteration_scheme=te_scheme,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            max_q_size=10,
            nb_val_examples=100,
            callbacks=[
                K.callbacks.TensorBoard(log_dir=log_dir),
                K.callbacks.CSVLogger(filename=log_dir + '/logger.csv'),
                K.callbacks.ModelCheckpoint(log_dir + '/model.h5'),
                ]
                ))
            )

    return model
