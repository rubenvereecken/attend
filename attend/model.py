import numpy as np
import tensorflow as tf

from attend.log import Log; log = Log.get_logger(__name__)
import attend
from attend import tf_util
from attend.attention import *
from functools import partial


class AttendModel():
    # ALLOWED_ATTN_IMPLS = ['attention']


    def __init__(self, provider, encoder, num_hidden=512, loss_fun='mse',
            time_steps=None, attention_impl=None, attention_units=None,
            dropout=None,
            use_dropout=True,
            final_activation=None,
            debug=True):
        """

        Arguments:
            time_steps: For BPTT, or None for dynamic LSTM (unimplemented)

        """

        self.provider = provider
        self.encoder = encoder

        self.dim_feature = provider.dim_feature
        self.D = np.prod(self.dim_feature) # Feature dimension when flattened
        self.H = num_hidden
        self.dropout = dropout
        self.use_dropout_for_training = use_dropout

        if isinstance(final_activation, str):
            self.final_activation = attend.get_activation(final_activation)
        else:
            self.final_activation = final_activation

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

        self.attention_units = attention_units
        self.attention_impl = attention_impl
        if attention_impl == 'bahdanau':
            self.attention_layer = partial(BahdanauAttention, attention_units, False)
        elif attention_impl.startswith('bahdanau_norm'):
            self.attention_layer = partial(BahdanauAttention, attention_units, True)
        elif attention_impl == 'time_naive':
            # Backward compatibility to rebuild old models
            self.attention_layer = attend.attention.OldTimeNaive
        elif attention_impl is None or attention_impl == 'none':
            self.attention_layer = None
        else:
            raise ValueError('Invalid attention impl {}'.format(attention_impl))

        self.debug = debug


    def build_model(self, provider, is_training=True):
        """Build the entire model

        Args:
            features: Feature batch Tensor (from provider)
            targets: Targets batch Tensor
        """
        features, targets = provider.features, provider.targets
        state_saver = provider.state_saver

        assert targets is None or targets.shape.ndims == 3, \
                'The target is assumed to be B x T x 1'

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

        x = features

        with tf.variable_scope('input_shape'):
            batch_size = tf.shape(features)[0]
            x = tf.reshape(x, [batch_size, -1, *self.dim_feature])

        x = self.encoder(x, provider, use_dropout)

        # TODO consider projecting features x

        if not state_saver is None:
            c = provider.state('lstm_c')
            h = provider.state('lstm_h')
            history = provider.state('history')
            last_out = provider.state('last_out')
            first = provider.state('first')
        else:
            c, h = self._initial_lstm(x)

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)

        outputs = []
        if self.attention_layer:
            attention = self.attention_layer()
            attentions = []
            alphas = []
        else:
            attentions = None
            alphas = None

        with tf.variable_scope('decoder'):
            lstm_scope = None
            decode_lstm_scope = None
            attention_scope = None
            for t in range(T):
                if t == 0:
                    # Get the last output from previous batch
                    true_prev_target = last_out
                elif is_training:
                    # Feed back in last true input
                    true_prev_target = targets[:, t-1]
                else:
                    # Feed back in previous prediction
                    true_prev_target = outputs[-1]


                if self.attention_layer:
                    # for t = 0, use current and t-1 from history
                    # for t = T-1, use all of current frame and none from history
                    with tf.variable_scope(attention_scope or 'attention', reuse=t!=0) as attention_scope:
                        past_window = tf.concat([history[:,t+1:,:], x[:,:t+1,:]], 1, name='window')
                        attention, alpha = attention(past_window, h, t!=0)
                        attentions.append(attention)
                        alphas.append(alpha)
                        decoder_lstm_input = tf.concat([true_prev_target, attention], 1)
                else:
                    decoder_lstm_input = tf.concat([true_prev_target, x[:,t,:]], 1)

                with tf.name_scope(lstm_scope or 'lstm') as lstm_scope:
                    # _ and h are the same, the LSTM output
                    _, (c, h) = lstm_cell(inputs=decoder_lstm_input, state=[c, h])

                with tf.name_scope(decode_lstm_scope or 'decode_lstm_step') as decode_lstm_scope:
                    output = self._decode(x[:,t,:], h, dropout=use_dropout, reuse=(t!=0))
                    outputs.append(output)

            outputs = tf.stack(outputs, axis=1) # B x T x 1
            if self.attention_layer:
                attentions = tf.stack(attentions, axis=1, name='attention')
                alphas = tf.stack(alphas, axis=1, name='alpha')

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
                    state_saver.save_state('history', x, 'history'),
                    state_saver.save_state('last_out', outputs[:,-1], 'last_out')
                )

                control_deps.append(save_state)

                # Makes it easier by just injecting the save state control op
                # into the rest of the computation graph, but also makes it messy
                with tf.control_dependencies(control_deps):
                    lengths = state_saver.length
                    outputs = tf.identity(outputs, name='output') # Tricksy way of injecting dependency
        else:
            assert len(control_deps) == 0

        context = {
                'length': lengths,
                'sequence_idx': state_saver.sequence,
                'sequence_count': state_saver.sequence_count,
                # 'original_key': original_keys,
                # 'key': state_saver.key,
                'key': self._extract_keys(state_saver.key),
                }
        tf_util.add_to_collection(attend.GraphKeys.CONTEXT, [*context.values()])

        assert outputs.shape.ndims == 3, 'B x T x 1'

        out = {
                'output': outputs,
                }
        tf_util.add_to_collection(attend.GraphKeys.OUTPUT, outputs)

        if self.attention_layer:
            out.update({
                'attention': attentions,
                'alpha': alphas,
                })
            tf_util.add_to_collection(attend.GraphKeys.OUTPUT, [attentions, alphas])

        return out, context


    def _extract_keys(self, keys):
        with tf.name_scope('extract_keys'):
            # Assume key looks like 'seqn:original_key:random', so 3 splits
            splits = tf.string_split(keys, ':')
            splits = tf.reshape(splits.values, ([-1,3])) # Reshape per key
            splits = splits[:,1:] # Drop the sequence info
            keys = tf.map_fn(lambda x: tf.string_join(tf.unstack(x), ':'), splits)
            keys = tf.identity(keys, name='key')
            return keys


    def calculate_loss(self, predictions, targets, lengths=None):
        # predictions.shape.assert_is_compatible_with(targets)

        with tf.variable_scope('loss'):
            T = tf.shape(targets)[1]
            # targets = tf.squeeze(targets) # Get rid of trailing 1-dimensions
            # Tails of sequences are likely padded, so create a mask to ignore padding
            mask = tf.cast(tf.sequence_mask(lengths, T), tf.int32) if lengths is not None else None
            if predictions.shape.ndims > 2:
                mask = tf.expand_dims(mask, -1)

            loss = self.loss_fun(targets, predictions, weights=mask)

        # if self.debug:
        #     loss = tf.Print(loss, [loss], message='loss ')

        return loss


    # It's important to have losses per entire sequence
    def calculate_losses(self, predictions, targets, keys, lengths=None, scope='loss'):
        from attend import metrics

        with tf.variable_scope(scope):
            if targets.shape.ndims > 2:
                targets = tf.squeeze(targets, axis=2, name='squeeze_target')
            if predictions.shape.ndims > 2:
                predictions = tf.squeeze(predictions, axis=2, name='squeeze_pred')
            shape = tf.shape(predictions)
            B, T = shape[0], shape[1]
            T = self.T
            length_table = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string,
                    value_dtype=tf.int64, default_value=0,
                    name='length_table')

            mask = tf.cast(tf.sequence_mask(lengths, T), tf.float32)
            # if predictions.shape.ndims > 2:
            #     mask = tf.expand_dims(mask, -1)
            length_update = length_table.insert(keys, tf.cast(lengths, tf.int64))

            out = dict(batch={}, all={}, total={})

            loss_tables = {}
            batch_losses = {}

            batch_mse, mse_table = metrics.streaming_mse(predictions, targets,
                    keys, lengths, mask)
            batch_losses['mse'] = batch_mse
            loss_tables['mse'] = mse_table

            corr, corr_update = tf.contrib.metrics.streaming_pearson_correlation(
                    predictions, targets, mask)
            tfmse, tfmse_update = tf.contrib.metrics.streaming_mean_squared_error(
                    predictions, targets, mask)

            error_updates = tf.group(corr_update, tfmse_update)

            with tf.control_dependencies([length_update, error_updates]):
                all_keys, all_lengths = length_table.export()
                out['context'] = { 'all_keys': all_keys,
                                   'all_lengths': all_lengths }
                for k in batch_losses.keys():
                    out['batch'][k] = batch_losses[k]
                    out['all'][k] = loss_tables[k].export()[1]

                out['total']['pearson_r'] = corr
                out['total']['mse_tf'] = tfmse

            return out


    def _decode(self, x, h, dropout=False, reuse=False):
        with tf.variable_scope('decode', reuse=reuse):
            decode_input = tf.concat([x, h], 1, name='decode_input')
            W = tf.get_variable('W', [decode_input.shape[1], 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            out = tf.matmul(h, W) + b
            out = self.final_activation(out)
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
