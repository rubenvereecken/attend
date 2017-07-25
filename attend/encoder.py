import tensorflow as tf
import numpy as np
import functools

from attend.log import Log; log = Log.get_logger(__name__)

class Encoder():
    ALLOWED_CONV_IMPLS = ['small', 'convnet', 'resnet', 'vggface', 'none']
    ACTIVATIONS = dict(relu=tf.nn.relu)

    def __init__(self, batch_size, encode_hidden_units=0, time_steps=None,
                 debug=True, conv_impl=None, dense_layer=0, dropout=.75,
                 dense_spec=None
                 ):
        self.debug = debug
        self.encode_lstm = encode_hidden_units > 0
        self.batch_size = batch_size # TODO this will go

        assert conv_impl in Encoder.ALLOWED_CONV_IMPLS

        if conv_impl is None and debug:
            self.conv_impl = 'small'
        elif conv_impl is None:
            self.conv_impl = 'convnet'
        else:
            self.conv_impl = conv_impl

        if encode_hidden_units: assert not time_steps is None, 'need time steps for encode lstm'
        self.encode_hidden_units = encode_hidden_units
        self.dense_layer         = dense_layer
        self.dense_spec = dense_spec
        self.time_steps          = time_steps
        self.T                   = time_steps
        self.dropout             = dropout

        # TODO check more thoroughly
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer  = tf.constant_initializer(0.0)


    def __call__(self, x, state_saver=None, use_dropout=True):
        with tf.variable_scope('encoder'):
            x = self.conv_network(x)
            if self.dense_layer:
                log.debug('Using a dense layer in the encoder')
                x = self.dense(x, use_dropout)
            elif self.dense_spec:
                log.debug('Building dense from spec `%s`', self.dense_spec)
                x = self.dense_from_spec(x, self.dense_spec, use_dropout)
            if self.encode_hidden_units > 0:
                x = self._encode_lstm(x, state_saver, use_dropout)
            return x


    def encoded_dim(self, input_dims):
        # If encoding with an LSTM, it's easy to figure out the size of the output beforehand
        if self.encode_lstm:
            return self.encode_hidden_units
        else:
            g = tf.Graph()
            with g.as_default():
                x = tf.placeholder(dtype=tf.float32, shape=[None, None, *input_dims])
                conv_out = self(x)
                log.debug('Encoder output shape %s', conv_out.shape)
                return conv_out.shape.as_list()[2:]


    def dense(self, x, use_dropout=True):
        # Expect flattened
        x.shape.assert_has_rank(3)
        D = x.shape[2]

        with tf.variable_scope('encode_dense'):
            x = tf.layers.dense(x, D, activation=tf.nn.relu,
                    kernel_initializer=self.weight_initializer,
                    name='dense')

            if use_dropout:
                x = tf.nn.dropout(x, self.dropout)

                # NOTE I'm not sure if maxnorm clipping doesn't blow it up
                # in case of null-states (like padded sequence tails)
                # because I don't care about those but we don't mask
                # TODO value 3 is from the original dropout paper I believe
            x = tf.clip_by_norm(x, 3)

        return x


    def dense_from_spec(self, x, fullspec, use_dropout=True):
        specs = fullspec.split(':')

        n = 0
        sizes = []
        for spec in specs:
            n += 1
            bits = spec.split(',')
            size = x.shape[-1] if bits[0] == '-' else int(bits[0])
            sizes.append(size)
            activation = Encoder.ACTIVATIONS[bits[1]] if bits[1] else None

            x = tf.layers.dense(x, size, activation=activation,
                    kernel_initializer=self.weight_initializer,
                    name='dense_{}'.format(n))
            if use_dropout:
                x = tf.nn.dropout(x, self.dropout, name='dropout_{}'.format(n))

            x = tf.clip_by_norm(x, 3, name='dropout_{}'.format(n))

        log.debug('Built %s dense layers sizes %s', n, sizes)

        return x


    def _conv2d(self, x, W, b, stride):
        x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        x = tf.nn.bias_add(x, b, name='conv2d_bias')
        x = tf.nn.relu(x)
        return x


    def _avg_pool(self, x, window, stride):
        return tf.nn.avg_pool(x, ksize=[1, window, window, 1],
                              strides=[1, stride, stride, 1], padding='SAME')


    def conv_network(self, x):
        if self.conv_impl == 'resnet':
            x = self._build_resnet(x)
        elif self.conv_impl == 'vggface':
            x = self._build_vggface(x)
        elif self.conv_impl == 'none':
            D_feat = x.shape[2:].as_list()
            D_feat = np.prod(D_feat)
            # Just flatten the features if no convolutional network defined
            return tf.reshape(x, [-1, self.time_steps, D_feat])
        else:
            x = self._build_conv_network(x)

        D_conv = x.shape[1:] # 14 x 14 x 512 for example

        with tf.name_scope('conv_reshape'):
            # This you would use for spatial attention
            # x = tf.reshape(x, [batch_size, -1, *D_conv.as_list()]) # B, T, D0, D1, D2
            # TODO consider stacked lstm
            # https://www.tensorflow.org/images/attention_seq2seq.png (tutorials/seq2seq)

            D_feat = np.prod(D_conv) # 14 x 14 x 512

            # Flatten conv2d output
            x = tf.reshape(x, [self.batch_size, self.time_steps, D_feat.value]) # B, T, 14*14*512

        print(x.shape)

        return x


    def _build_resnet(self, x, out_layer='avg_pool'):
        import tensorflow.contrib.keras as K
        K.backend.set_learning_phase(True) # TODO change to 0 for test

        dim_feature = tuple(x.shape.as_list()[2:])
        with tf.name_scope('conv_reshape'):
            x = tf.reshape(x, [-1, *dim_feature])

        resnet = K.applications.ResNet50(
                include_top=False,
                weights='imagenet',
                # input_tensor=x,
                input_shape=dim_feature)

        out_layer = resnet.get_layer(out_layer).output
        resnet_conv = K.models.Model(resnet.input, out_layer)
        # resnet.summary()
        out = resnet_conv(x)
        return out


    def _build_vggface(self, x, out_layer='global_average_pooling2d_1'):
        """
        out_layer: 'pool5' for 7x7x512
                   'global_average_pooling2d_1' for x512 (flat)
        """
        import tensorflow.contrib.keras as K
        from keras_vggface.vggface import VGGFace
        K.backend.set_learning_phase(True) # TODO change to 0 for test

        dim_feature = tuple(x.shape.as_list()[2:])
        with tf.name_scope('conv_reshape'):
            x = tf.reshape(x, [-1, *dim_feature])

        vgg_model = VGGFace(include_top=False, input_shape=dim_feature,
                            pooling='avg')
        out_layer = vgg_model.get_layer(out_layer).output
        vgg_conv = K.models.Model(vgg_model.input, out_layer)
        out = vgg_conv(x)

        return out # 7 x 7 x 512 or None x 512


    def _build_conv_network(self, x):
        """ConvNet: 5 conv layers, 3 avg pooling layers
        Arguments
            x: input Tensor (B, T, D0,...)

        Out
            x: output Tensor (T, 14, 14, 512)
        """
        import time
        start = time.time()
        n_channels = x.shape[-1]
        if self.conv_impl == 'convnet':
            self.conv_filters = [
                [3, 3, n_channels.value, 96],
                [3, 3, 96, 256],
                [3, 3, 256, 512],
                [3, 3, 512, 512],
                [3, 3, 512, 512]
            ]
            self.pool_windows = [3, 3, 0, 0, 3]
            # pool_windows = [3, 3, 3, 0, 3]
            self.conv_strides = [1, 2, 1, 1, 1]
            self.pool_strides = [2, 2, 0, 0, 2]
            # pool_strides = [2, 2, 2, 0, 2]
        elif self.conv_impl == 'small':
            self.conv_filters = [
                [3, 3, n_channels.value, 96],
                [3, 3, 96, 128],
                [3, 3, 128, 256],
                [3, 3, 256, 256],
            ]
            self.pool_windows = [3, 3, 0, 3]
            self.conv_strides = [2, 2, 1, 1]
            self.pool_strides = [2, 2, 2, 2]
        else:
            raise Exception('Unkown conv impl {}'.format(self.conv_impl))

        dim_feature = x.shape.as_list()[2:]

        with tf.name_scope('conv_reshape'):
            x = tf.reshape(x, [-1, *dim_feature])

        with tf.variable_scope('ConvNet'):

            for i in range(len(self.conv_filters)):
                with tf.variable_scope('conv{}'.format(i)):
                    W = tf.Variable(self.weight_initializer(self.conv_filters[i]), name='W')
                    b = tf.Variable(self.weight_initializer([self.conv_filters[i][-1]]), name='b')

                    # Apply 2D convolution
                    x = self._conv2d(x, W, b, stride=self.conv_strides[i])

                # Apply avg pooling if required
                if self.pool_windows[i]:
                    with tf.variable_scope('pool{}'.format(i)):
                        x = self._avg_pool(x, self.pool_windows[i], self.pool_strides[i])

            if self.debug:
                x = tf.Print(x, [tf.shape(x)[1:]], message='Conv2d output shape ')
            # Fully connected layer
            # Reshape conv2 output to fit fully connected layer input
            # TODO change this once attention on images in place
            # W = tf.Variable(self.weight_initializer([512, 512]))
            # b = tf.Variable(self.weight_initializer([512]))
            # x = tf.einsum('ijkl,lm->ijkm', x, W)
            # x = tf.nn.bias_add(x, b, name='fc_bias')

        print('Built convnet in {:.3f}s'.format(time.time()-start))

        return x


    def _encode_lstm(self, x, state_saver=None, use_dropout=True):
        from attend.provider import Provider

        with tf.variable_scope('encode_lstm'):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.encode_hidden_units)
            x_by_time = tf.split(value=x, num_or_size_splits=self.time_steps, axis=1)
            x_by_time = list(map(lambda x: tf.squeeze(x, axis=1), x_by_time))
            # assert len(x_by_time) == self.time_steps

            if not state_saver is None:
                out, _ = tf.contrib.rnn.static_state_saving_rnn(lstm_cell, x_by_time,
                        state_saver=state_saver,
                        state_name=(Provider.ENCODE_LSTM_C, Provider.ENCODE_LSTM_H))
            else:
                # Dynamic because we assume difference time lengths come through
                out, _ = tf.nn.dynamic_rnn(lstm_cell, x_by_time)


            # TODO for some weird reason their implementation needs a batch size
            # So let's do our own once the time's there
            out = tf.stack(out, axis=1)

            return out
