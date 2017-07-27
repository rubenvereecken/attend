import numpy as np
import tensorflow as tf

from attend.log import Log; log = Log.get_logger(__name__)


class AttendModel():
    # ALLOWED_ATTN_IMPLS = ['attention']


    def __init__(self, provider, encoder, num_hidden=512, loss_fun='mse',
            time_steps=None, attention_impl=None,
            dropout=None,
            use_dropout=True,
            debug=True):
        """

        Arguments:
            time_steps: For BPTT, or None for dynamic LSTM (unimplemented)

        """

        self.provider = provider
        self.encoder = encoder

        # self.batch_size = batch_size
        # self.dim_feature = provider.dim_feature
        # self.dim_feature = [224, 224, 3]
        self.dim_feature = provider.dim_feature
        self.D = np.prod(self.dim_feature) # Feature dimension when flattened
        self.H = num_hidden
        self.dropout = dropout
        self.use_dropout_for_training = use_dropout

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

        self.attention_impl = attention_impl
        if attention_impl == 'time_naive':
            self.attention_layer = self._attention_layer
        elif attention_impl is None or attention_impl == 'none':
            self.attention_layer = None
        else:
            raise ValueError('Invalid attention impl {}'.format(attention_impl))

        self.debug = debug


    def _attention_layer(self, x, h, reuse=False):
        """
        h: decoder hidden state from previous step
        """
        # Require features to be flat at this point
        x.shape.assert_has_rank(3)

        with tf.variable_scope('project_features', reuse=reuse):
            W = tf.get_variable('W', [self.D, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            feat_proj = tf.einsum('ijk,kl->ijl', x, W) + b

        with tf.variable_scope('attention_layer', reuse=reuse):
            # TODO dropout on attention hidden weights
            W = tf.get_variable('W', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            h_att = tf.nn.relu(feat_proj + tf.expand_dims(tf.matmul(h, W), 1) + b, name='h_att')

            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)
            out_att = tf.einsum('ijk,kl->ij', h_att, w_att)
            # Softmax assigns probability to each frame
            alpha = tf.nn.softmax(out_att, name='alpha')
            context = tf.reduce_sum(x * tf.expand_dims(alpha, 2), 1, name='context')

            return context, alpha


    def build_model(self, provider, is_training=True):
        """Build the entire model

        Args:
            features: Feature batch Tensor (from provider)
            targets: Targets batch Tensor
        """
        features, targets = provider.features, provider.targets
        state_saver = provider.state_saver

        use_dropout = is_training and self.use_dropout_for_training
        if is_training:
            log.debug('=== TRAINING ===')
        else:
            log.debug('=== VALIDATION ===')
        if use_dropout:
            log.debug('Using dropout %s', self.dropout)
        else:
            log.debug('NO dropout used')

        if use_dropout and self.dropout in [1, None]:
            raise ValueError('Bad dropout value %s', self.dropout)

        T = self.T if not self.T is None else tf.shape(features[1])

        # Consider batch normalizing features instead of mean subtract
        x = features

        with tf.variable_scope('input_shape'):
            batch_size = tf.shape(features)[0]
            x = tf.reshape(x, [batch_size, -1, *self.dim_feature])
            # if self.debug:
            #     x = tf.Print(x, [tf.shape(features)], message='Input feat shape ')

        x = self.encoder(x, state_saver, use_dropout)
        log.debug('encoded shape %s', x.shape)

        # TODO consider projecting features x

        if not state_saver is None:
            c = state_saver.state('lstm_c')
            h = state_saver.state('lstm_h')
            history = state_saver.state('history')
        else:
            c, h = self._initial_lstm(x)

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)

        outputs = []

        with tf.variable_scope('decoder'):
            lstm_scope = None
            decode_lstm_scope = None
            for t in range(T):
                if t == 0:
                    # Fill with null values first go because there is no previous
                    true_prev_target = tf.zeros([batch_size, 1]) # T x 1
                elif is_training:
                    # Feed back in last true input
                    true_prev_target = targets[:, t-1]
                else:
                    # Feed back in previous prediction
                    true_prev_target = output


                if self.attention_layer:
                    # for t = 0, use current and t-1 from history
                    # for t = T-1, use all of current frame and none from history
                    past_window = tf.concat([history[:,t+1:,:], x[:,:t+1,:]], name='window')
                    log.debug('Enabling attention with a %s step window', T)
                    context, alpha = self.attention_layer(x, h)
                    decoder_lstm_input = tf.concat([true_prev_target, context], 1)
                else:
                    decoder_lstm_input = tf.concat([true_prev_target, x[:,t,:]], 1)

                # TODO bring scope outside?
                with tf.name_scope(lstm_scope or 'lstm') as lstm_scope:
                    _, (c, h) = lstm_cell(inputs=decoder_lstm_input, state=[c, h])

                with tf.name_scope(decode_lstm_scope or 'decode_lstm_step') as decode_lstm_scope:
                    output = self._decode(c, dropout=True, reuse=(t!=0))
                    outputs.append(output)
            outputs = tf.squeeze(tf.stack(outputs, axis=1), axis=[2]) # B x T

        control_deps = []

        # Saves LSTM state so decoding can continue like normal in a next call
        if not state_saver is None:
            save_state_scope = 'save_state'
            with tf.variable_scope(save_state_scope):
                save_state = tf.group(
                    state_saver.save_state('lstm_c', c, 'lstm_c'),
                    state_saver.save_state('lstm_h', h, 'lstm_h'),
                    state_saver.save_state('first',
                        tf.zeros([batch_size], dtype=tf.bool), 'first'),
                    state_saver.save_state('history', x, 'history')
                )
            control_deps.append(save_state)

        # Makes it easier by just injecting the save state control op
        # into the rest of the computation graph, but also makes it messy
        with tf.control_dependencies(control_deps):
            lengths = state_saver.length
            outputs = tf.identity(outputs) # Tricksy way of injecting dependency

        context = {
                'length': lengths,
                'sequence_idx': state_saver.sequence,
                'sequence_count': state_saver.sequence_count,
                # 'original_key': original_keys,
                # 'key': state_saver.key,
                'key': self._extract_keys(state_saver.key),
                }

        return outputs, context


    def _extract_keys(self, keys):
        # Assume key looks like 'seqn:original_key:random', so 3 splits
        splits = tf.string_split(keys, ':')
        splits = tf.reshape(splits.values, ([-1,3])) # Reshape per key
        splits = splits[:,1:] # Drop the sequence info
        keys = tf.map_fn(lambda x: tf.string_join(tf.unstack(x), ':'), splits)
        return keys


    def calculate_loss(self, predictions, targets, lengths=None):
        with tf.variable_scope('loss'):
            T = tf.shape(targets)[1]
            targets = tf.squeeze(targets) # Get rid of trailing 1-dimensions
            # Tails of sequences are likely padded, so create a mask to ignore padding
            mask = tf.cast(tf.sequence_mask(lengths, T), tf.int32) if lengths is not None else None
            loss = self.loss_fun(targets, predictions, weights=mask)

        # if self.debug:
        #     loss = tf.Print(loss, [loss], message='loss ')

        return loss

    def streaming_mse(self, predictions, targets, keys, lengths):
        T = self.T

        with tf.name_scope('streaming_mse'):
            mask = tf.cast(tf.sequence_mask(lengths, T), tf.float32)

            totals = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string,
                        value_dtype=tf.float32, default_value=0,
                        name='total_table')
            counts = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string,
                        value_dtype=tf.float32, default_value=0,
                        name='count_table')
            mse = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string,
                        value_dtype=tf.float32, default_value=0,
                        name='mse_table')

            total_se = totals.lookup(keys)
            masked_diff = (predictions - targets) * mask
            current_se = tf.reduce_sum(tf.square(masked_diff, name='mse'), axis=1)
            total_se += current_se
            total_update = totals.insert(keys, total_se)
            count = counts.lookup(keys)
            count += tf.cast(lengths, tf.float32)
            count_update = counts.insert(keys, count)
            mse_update = mse.insert(keys, total_se / (count))

        with tf.control_dependencies([total_update, count_update, mse_update]):
            return mse.lookup(keys), mse


    # It's important to have losses per entire sequence
    def calculate_losses(self, predictions, targets, keys, lengths=None, scope='loss'):
        with tf.variable_scope(scope):
            targets = tf.squeeze(targets, axis=2)
            shape = tf.shape(predictions)
            B, T = shape[0], shape[1]
            T = self.T
            length_table = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string,
                    value_dtype=tf.int64, default_value=0,
                    name='length_table')

            # mask = tf.cast(tf.sequence_mask(lengths, T), tf.float32)
            length_update = length_table.insert(keys, tf.cast(lengths, tf.int64))

            out = dict(batch={}, all={})

            loss_tables = {}
            batch_losses = {}

            batch_mse, mse_table = self.streaming_mse(predictions, targets,
                    keys, lengths)
            batch_losses['mse'] = batch_mse
            loss_tables['mse'] = mse_table

            with tf.control_dependencies([length_update]):
                all_keys, all_lengths = length_table.export()
                out['context'] = { 'all_keys': all_keys,
                                   'all_lengths': all_lengths }
                for k in batch_losses.keys():
                    out['batch'][k] = batch_losses[k]
                    out['all'][k] = loss_tables[k].export()[1]

            return out


    def _decode(self, h, dropout=False, reuse=False):
        with tf.variable_scope('decode', reuse=reuse):
            W = tf.get_variable('W', [h.shape[1], 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            out = tf.matmul(h, W) + b
            # out = tf.nn.sigmoid(tf.matmul(h, W) + b)
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
