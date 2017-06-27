import tensorflow as tf
# import tensorflow.contrib.keras as K
# from tensorflow.contrib.keras import applications as apps

# from fuel.datasets.hdf5 import H5PYDataset
# from fuel.streams import DataStream

# from callbacks import TensorBoard
# from schemes import InfiniteSequentialBatchIterator as InfSeqBatchIterator


class AttendModel():
    def __init__(self, batch_size=None, debug=True):

        # self.batch_size = batch_size
        self.dim_feature = (224, 224, 3)
        self.n_channels = self.dim_feature[2]
        self.num_steps = 20

        # self.features = tf.placeholder(tf.float32, [batch_size, *dim_feat])
        # self.targets = tf.placeholder(tf.float32, [batch_size, n_time_step])
        self.conv_weight_initializer = tf.random_normal

        self.debug = debug


    # TODO split this up in predict and loss or something
    def build_model(self, features, targets):
        """Build the entire model

        Args:
            features: Feature batch Tensor (from provider)
            targets: Targets batch Tensor
        """

        batch_size = tf.shape(targets)[0]
        T = tf.shape(targets)[1]

        x = features
        x = tf.reshape(x, [batch_size, -1, *self.dim_feature])
        if self.debug:
            x = tf.Print(x, [tf.shape(features)], message='Input feat shape ')
        x = tf.reshape(x, [-1, *self.dim_feature])
        x = self._conv_network(x)
        # x = tf.reshape(x, [batch_size, -1, *self.dim_feature])

        # Next up
        # Write a BTT LSTM, for `self.time_steps`
        # Base self on static RNN
        # Look at what TFs rnn wrappers do
        # Don't forget the pretty TensorBoard graph for inspiration
        # I don't think dynamic LSTMs are needed at this point

        return tf.reduce_sum(tf.slice(x, [0, 0, 0, 0], [0, 0, 0, 1]))


    def _conv2d(self, x, W, b, stride):
        x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        x = tf.nn.bias_add(x, b, name='conv2d_bias')
        x = tf.nn.relu(x)
        return x


    def _avg_pool(self, x, window, stride):
        return tf.nn.avg_pool(x, ksize=[1, window, window, 1],
                              strides=[1, stride, stride, 1], padding='SAME')


    def _conv_network(self, x):
        """ConvNet: 5 conv layers, 3 avg pooling layers

        Out
            x: output Tensor (T, 14, 14, 512)
        """
        conv_filters = [
            [3, 3, self.n_channels, 96],
            [3, 3, 96, 256],
            [3, 3, 256, 512],
            [3, 3, 512, 512],
            [3, 3, 512, 512]
        ]
        pool_windows = [3, 3, 0, 0, 3]
        conv_strides = [1, 2, 1, 1, 1]
        pool_strides = [2, 2, 0, 0, 2]

        for i in range(len(conv_filters)):
            W = tf.Variable(self.conv_weight_initializer(conv_filters[i]))
            b = tf.Variable(self.conv_weight_initializer([conv_filters[i][-1]]))

            # Apply 2D convolution
            x = self._conv2d(x, W, b, stride=conv_strides[i])

            # Apply avg pooling if required
            if pool_windows[i]:
                x = self._avg_pool(x, pool_windows[i], pool_strides[i])

        if self.debug:
            x = tf.Print(x, [tf.shape(x)], message='Conv2d output shape ')
        # Fully connected layer
	# Reshape conv2 output to fit fully connected layer input
        # TODO change this once attention on images in place
        # W = tf.Variable(self.conv_weight_initializer([]))
	# x = tf.reshape(x, [-1, weights['wd1'].get_shape().as_list()[0]])
	# fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	# fc1 = tf.nn.relu(fc1)
	# # Apply Dropout
	# fc1 = tf.nn.dropout(fc1, dropout)
        W = tf.Variable(self.conv_weight_initializer([512, 512]))
        b = tf.Variable(self.conv_weight_initializer([512]))
        # TODO it's fine switching them right
        x = tf.einsum('ijkl,lm->ijkm', x, W)
        # x = tf.reshape(x, [-1, 512])
        # x = tf.matmul(x, W)
        x = tf.nn.bias_add(x, b, name='fc_bias')

        return x






# def linear_reg(nb_outputs, dropout=.5, norm=True, sizes=[512, 512]):

#     # Fit the Keras way of doing thangs
#     def _network(net):
#         for size in sizes: net = K.layers.Dense(size)(net)
#             net = K.layers.BatchNormalization(axis=-1)(net)
#             net = K.layers.Dropout(dropout)(net)

#         net = K.layers.Dense(nb_outputs, activation='softmax')(net)
#         return net

#     return _network


# def _target_path_for(target):
#     return '/{}/V/aligned'.format(target)


# def setup_model(
#         target,
#         data_file,
#         batch_size,
#         steps_per_epoch,
#         epochs,
#         log_dir
#         ):
#     # Define the datasets
#     # Select the features and a single target
#     sources = ['img', _target_path_for(target)]
#     tr_set = H5PYDataset(data_file, which_sets=('train',), sources=sources)
#     te_set = H5PYDataset(data_file, which_sets=('test',), sources=sources)

#     # I feel like this can be done better
#     tr_scheme = InfSeqBatchIterator(examples=tr_set.num_examples,
#             batch_size=batch_size)
#     te_scheme = InfSeqBatchIterator(examples=te_set.num_examples,
#             batch_size=batch_size)

#     tr_stream = DataStream(dataset=tr_set, iteration_scheme=tr_scheme)
#     te_stream = DataStream(dataset=te_set, iteration_scheme=te_scheme)

#     # input_layer = K.layers.Input(tr_set.source_shapes[0].shape[1:])
#     base_model = apps.ResNet50(weights = 'imagenet')
#     input_layer = base_model.input
#     last_layer = base_model.get_layer('flatten_1').output
#     pred = linear_reg(1)(last_layer)

#     model = K.models.Model([input_layer], pred)

#     model.compile(
#             optimizer = K.optimizers.Adadelta(
#                 lr = 1.,
#                 rho = .95,
#                 epsilon = 1e-8,
#                 decay = 5e-5,
#                 ),
#             loss = K.losses.mean_squared_error
#             )

#     model.fit_generator(
#             generator=tr_stream.get_epoch_iterator(),
#             steps_per_epoch=steps_per_epoch,
#             epochs=epochs,
#             max_q_size=10,
#             # nb_val_samples=100,
#             validation_data=te_stream.get_epoch_iterator(),
#             validation_steps=100,
#             callbacks=[
#                 # K.callbacks.TensorBoard(log_dir=log_dir),
#                 TensorBoard(log_dir=log_dir),
#                 K.callbacks.CSVLogger(filename=log_dir + '/logger.csv'),
#                 K.callbacks.ModelCheckpoint(log_dir + '/model.h5'),
#                 ]
#             )

#     return model
