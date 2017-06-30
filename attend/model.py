import numpy as np
import tensorflow as tf


class AttendModel():
    def __init__(self, provider, encoder, num_hidden=512, loss_fun='mse', time_steps=None, debug=True):
        """

        Arguments:
            time_steps: For BPTT, or None for dynamic LSTM (unimplemented)

        """

        self.provider = provider
        self.encoder = encoder

        # self.batch_size = batch_size
        # self.dim_feature = provider.dim_feature
        self.dim_feature = [224, 224, 3]
        self.n_channels = self.dim_feature[2]
        # self.num_steps = 20
        self.H = num_hidden

        # If None, it's variable, if an int, it's known. Can also be Tensor
        self.T = time_steps
        assert not self.T is None, "Remove this once other feat implemented"

        # self.features = tf.placeholder(tf.float32, [batch_size, *dim_feat])
        # self.targets = tf.placeholder(tf.float32, [batch_size, n_time_step])
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)

        if loss_fun == 'mse':
            self.loss_fun = tf.losses.mean_squared_error
        else:
            raise Exception()

        self.debug = debug


    # TODO split this up in predict and loss or something
    def build_model(self, provider, seq_q=True):
        """Build the entire model

        Args:
            features: Feature batch Tensor (from provider)
            targets: Targets batch Tensor
        """
        assert seq_q

        features, targets = provider.features, provider.targets
        state_saver = provider.state_saver

        batch_size = tf.shape(targets)[0]
        T = self.T if not self.T is None else tf.shape(targets[1])

        # TODO consider batch normalizing features instead of mean subtract
        x = features
        x = tf.reshape(x, [batch_size, -1, *self.dim_feature])
        if self.debug:
            x = tf.Print(x, [tf.shape(features)], message='Input feat shape ')
        x = self.encoder(x, state_saver)
        print(x.shape)

        # c, h = self._initial_lstm(x)
        c = state_saver.state('lstm_c')
        h = state_saver.state('lstm_h')
        history = state_saver.state('history')

        # TODO that other implementation projects the features first
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)

        outputs = []

        with tf.variable_scope('decoder'):
            for t in range(T):
                # TODO this will become attention
                context = x[:,t,:]

                if t == 0:
                    # Fill with null values first go because there is no previous
                    # Assumes target dim of 1 per time step
                    true_prev_target = tf.zeros([batch_size, 1])
                else:
                    # Feed back in last true input
                    # print(targets.shape)
                    true_prev_target = targets[:, t]

                # TODO bring scope outside?
                with tf.name_scope('lstm'):
                    _, (c, h) = lstm_cell(inputs=tf.concat([true_prev_target, context], 1),
                            state=[c, h])

                with tf.name_scope('decode_lstm'):
                    output = self._decode(c, dropout=True, reuse=(t!=0))
                    outputs.append(output)

        # Saves LSTM state so decoding can continue like normal in a next call
        with tf.name_scope('save_state'):
            save_state = tf.group(
                    state_saver.save_state('lstm_c', c),
                    state_saver.save_state('lstm_h', h)
            )
        # Makes it eaoutputssier by just injecting the save state control op
        # into the rest of the computation graph, but also makes it messy
        # TODO execute it separately
        with tf.control_dependencies([save_state]):
            with tf.variable_scope('loss'):
                outputs = tf.squeeze(tf.stack(outputs, axis=1)) # B x T
                # TODO squeeze elsewhere man
                targets = tf.squeeze(targets)
                # outputs = tf.Print(outputs, [tf.shape(outputs)], message='outputs shape ')
                # targets = tf.Print(targets, [tf.shape(targets)], message='targets shape ')

                # Tails of sequences are likely padded, so create a mask to ignore padding
                mask = tf.cast(tf.sequence_mask(state_saver.length, T), tf.int32)
                loss = self.loss_fun(targets, outputs, weights=mask)

        loss = tf.Print(loss, [loss], message='loss ')

        return loss


    def _decode(self, h, dropout=False, reuse=False):
        with tf.variable_scope('decode', reuse=reuse):
            W = tf.get_variable('W', [h.shape[1], 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            out = tf.nn.sigmoid(tf.matmul(h, W) + b)
            return out


    def _initial_lstm(self, features, reuse=False):
        """Initialize LSTM cell and hidden
        h = mean(x) * w_h + b_h
        c = mean(x) * w_c + b_c
        """
        D = np.prod(features.shape[2:])
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
        return c, h








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
