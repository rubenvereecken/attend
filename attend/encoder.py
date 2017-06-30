import tensorflow as tf
import numpy as np


class Encoder():
    def __init__(self, batch_size, encode_hidden_units=0, time_steps=None, debug=True, conv_impl=None):
        self.debug = debug
        self.encode_lstm = encode_hidden_units > 0
        self.batch_size = batch_size # TODO this will go

        if conv_impl is None and debug:
            self.conv_impl = 'small'
        elif conv_impl is None:
            self.conv_impl = 'convnet'
        else:
            self.conv_impl = conv_impl

        if encode_hidden_units: assert not time_steps is None, 'need time steps for encode lstm'
        self.encode_hidden_units = encode_hidden_units
        self.time_steps          = time_steps
        self.T                   = time_steps

        # Maybe look into this one
        self.weight_initializer = tf.random_normal


    def __call__(self, x, state_saver=None):
        with tf.variable_scope('encoder'):
            x = self.conv_network(x)
            if self.encode_hidden_units > 0:
                x = self._encode_lstm(x, state_saver)
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
                print(conv_out.shape)
                return conv_out.shape.as_list()[1:]


    def _conv2d(self, x, W, b, stride):
        x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        x = tf.nn.bias_add(x, b, name='conv2d_bias')
        x = tf.nn.relu(x)
        return x


    def _avg_pool(self, x, window, stride):
        return tf.nn.avg_pool(x, ksize=[1, window, window, 1],
                              strides=[1, stride, stride, 1], padding='SAME')


    def conv_network(self, x):
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
            self.pool_windows = [4, 4, 3, 3]
            self.conv_strides = [2, 2, 1, 1]
            self.pool_strides = [2, 2, 2, 2]
        else:
            raise Exception('Unkown conv impl {}'.format(self.conv_impl))

        dim_feature = x.shape.as_list()[2:]

        with tf.variable_scope('ConvNet'):
            x = tf.reshape(x, [-1, *dim_feature])

            for i in range(len(self.conv_filters)):
                with tf.variable_scope('conv{}'.format(i)):
                    W = tf.Variable(self.weight_initializer(self.conv_filters[i]), name='W')
                    b = tf.Variable(self.weight_initializer([self.conv_filters[i][-1]]), name='b')

                    # Apply 2D convolution
                    x = self._conv2d(x, W, b, stride=self.conv_strides[i])
                    # print(x.shape)

                # Apply avg pooling if required
                if self.pool_windows[i]:
                    with tf.variable_scope('pool{}'.format(i)):
                        x = self._avg_pool(x, self.pool_windows[i], self.pool_strides[i])
                        # print(x.shape)

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

        D_conv = x.shape[1:] # 14 x 14 x 512

        with tf.name_scope('conv_reshape'):
            # This you would use for spatial attention
            # x = tf.reshape(x, [batch_size, -1, *D_conv.as_list()]) # B, T, D0, D1, D2
            # TODO consider stacked lstm
            # https://www.tensorflow.org/images/attention_seq2seq.png (tutorials/seq2seq)

            D_feat = np.prod(D_conv) # 14 x 14 x 512

            # Flatten conv2d output
            x = tf.reshape(x, [self.batch_size, self.time_steps, D_feat.value]) # B, T, 14*14*512

        return x


    def _encode_lstm(self, x, state_saver=None):
        from attend.provider import Provider

        with tf.variable_scope('encode_lstm'):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.encode_hidden_units)
            x_by_time = tf.split(value=x, num_or_size_splits=self.time_steps, axis=1)
            x_by_time = list(map(tf.squeeze, x_by_time))
            # assert len(x_by_time) == self.time_steps
            # Could have used `static_state_saving_rnn` but
            # c = state_saver.state('encoder_lstm_c')
            out, _ = tf.contrib.rnn.static_state_saving_rnn(lstm_cell, x_by_time,
                    state_saver=state_saver,
                    state_name=(Provider.ENCODE_LSTM_C, Provider.ENCODE_LSTM_H))
            # TODO for some weird reason their implementation needs a batch size
            # So let's do our own once the time's there
            out = tf.stack(out, axis=1)

            return out
