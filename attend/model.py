import numpy as np
import tensorflow as tf

from attend.log import Log
import attend
from attend import tf_util
from attend.attention import BahdanauAttention
from attend.sample import get_sampler
from attend.common import get_activation, get_loss_function
from functools import partial

log = Log.get_logger(__name__)


class AttendModel():
    # ALLOWED_ATTN_IMPLS = ['attention']

    def __init__(self, provider, encoder, num_hidden=512, loss_function='mse',
                 time_steps=None, attention_impl=None, attention_units=None,
                 dropout=None,
                 use_dropout=True,
                 use_batch_norm=True,
                 batch_norm_decay=None,
                 use_batch_renorm=False,
                 final_activation=None,
                 sampling_scheme=None,
                 sampling_min=.75,  # 75% chance to pick truth
                 sampling_decay_steps=None,
                 sampler_kwargs={}, # Just for FixedEpsilonSampler
                 attention_input=None,
                 attention_score_nonlinearity=None,
                 num_image_patches=None,
                 # Variations to try
                 enc2lstm=False, enc2ctx=False,
                 debug=True):
        """
        Arguments:
            time_steps: For BPTT, or None for dynamic LSTM (unimplemented)

        """
        self.sampler = get_sampler(sampling_scheme)(sampling_min, **sampler_kwargs)
        self.sampling_decay_steps = sampling_decay_steps

        self.enc2lstm = enc2lstm
        self.enc2ctx = enc2ctx
        self._sampling_min = sampling_min

        self.provider = provider
        self.encoder = encoder

        self.dim_feature = provider.dim_feature
        self.D = np.prod(self.dim_feature)  # Feature dimension when flattened
        self.H = num_hidden
        self.L = num_image_patches # None if just flat features, used for attn
        self.dropout = dropout
        self.use_dropout_for_training = use_dropout
        self.use_batch_norm = use_batch_norm
        self.batch_norm_decay = batch_norm_decay
        self.use_batch_renorm = use_batch_renorm

        if isinstance(final_activation, str):
            self.final_activation = attend.get_activation(final_activation)
        else:
            self.final_activation = final_activation

        # If None, it's variable, if an int, it's known. Can also be Tensor
        self.T = time_steps
        assert self.T is not None, "Remove this once other feat implemented"

        # self.features = tf.placeholder(tf.float32, [batch_size, *dim_feat])
        # self.targets = tf.placeholder(tf.float32, [batch_size, n_time_step])
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)

        self.loss_fun = get_loss_function(loss_function)

        if attention_input is None: attention_input = 'time'
        self.attention_input = attention_input
        self.attention_units = attention_units
        self.attention_impl = attention_impl
        if attention_impl is None or attention_impl == 'none':
            self.attention_layer = None
        elif attention_impl == 'bahdanau':
            self.attention_layer = partial(
                BahdanauAttention, attention_units, False,
                attention_score_nonlinearity)
        elif attention_impl.startswith('bahdanau_norm'):
            self.attention_layer = partial(
                BahdanauAttention, attention_units, True,
                attention_score_nonlinearity)
        elif attention_impl == 'time_naive':
            # Backward compatibility to rebuild old models
            self.attention_layer = attend.attention.OldTimeNaive
        else:
            raise ValueError('Invalid attention impl {}'.format(attention_impl))

        self.debug = debug

    def build_model(self, provider, is_training, total_steps=None):
        """Build the entire model

        Args:
            provider: Provider, has `features` and `targets`
            is_training: bool
            total_steps: For annealing shemes
        """
        features, targets = provider.features, provider.targets
        state_saver = provider.state_saver

        if not self.sampling_decay_steps is None:
            log.debug('Using sampling decay steps from configuration')
            total_steps = self.sampling_decay_steps

        if is_training and total_steps is None:
            log.warning('total_steps=None but is_training=True')
            log.warning(' -- should only be the case when using FixedScheduler')
            # assert not total_steps is None, "Need total steps for scheduled sampling"

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

        T = self.T if self.T is not None else tf.shape(features[1])

        x = features

        with tf.variable_scope('input_shape'):
            batch_size = tf.shape(features)[0]
            x = tf.reshape(x, [batch_size, -1, *self.dim_feature])

        x = self.encoder(x, provider, is_training)

        if state_saver is not None:
            c = provider.state('lstm_c')
            h = provider.state('lstm_h')
            history = provider.state('history')
            context = provider.state('context')
            output = provider.state('output')
            target = provider.state('target')
            first = provider.state('first')
        else:
            c, h = self._initial_lstm(x)

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)

        outputs = []
        if self.attention_layer:
            do_attention = self.attention_layer()
            attentions = []
            alphas = []
        else:
            attentions = None
            alphas = None

        with tf.variable_scope('decoder'):
            lstm_scope = None
            decode_lstm_scope = None
            attention_scope = None
            attention_input_scope = None
            sample_scope = None

            # TODO refactor total_steps to elsewhere without breaking my
            # previous paradigms. Would make the log.warning above unneeded
            self.sampler.prepare(total_steps, is_training)

            for t in range(T):
                # y_t-1
                with tf.name_scope(sample_scope or 'sample') as sample_scope:
                    if t != 0:
                        target = targets[:,t-1]

                    # Always last output if not is_training
                    prev_target = self.sampler.sample(target, output, is_training)

                # if is_training and t == 0:
                    # This would be the naive case, learn two input values

                    # NOTE disabled for now because it looks like a lot of
                    # trouble for barely anything in return
                    # If not first, then this is just normal sampling
                    # If it is first, use the learned 'output' value
                    # def _sample_single(i):
                    #     # Turn into singleton batches then squeeze
                    #     return lambda: tf.squeeze(self._sample_output(
                    #         tf.expand_dims(target[i], 0),
                    #         tf.expand_dims(output[i], 0), epsilon), 0)

                    # def _sample_unless_first(i):
                    #     return tf.cond(first[i],
                    #                    true_fn=lambda: output[i],
                    #                    false_fn=_sample_single(i))

                    # prev_target = tf.map_fn(_sample_unless_first,
                    #                         tf.range(batch_size),
                    #                         dtype=tf.float32)

                with tf.name_scope(lstm_scope or 'lstm') as lstm_scope:
                    # Context is just previous encoded when not using attention
                    decoder_lstm_input = tf.concat(
                        [prev_target, context],
                        1, 'decoder_lstm_input')
                    if self.enc2lstm:
                        decoder_lstm_input = tf.concat(
                            [decoder_lstm_input, x[:, t, :]], 1)

                    # _ and h are the same, the LSTM output
                    _, (c, h) = lstm_cell(
                        inputs=decoder_lstm_input, state=[c, h])

                if self.attention_layer and self.attention_input == 'time':
                    # for t = 0, use current and t-1 from history
                    # for t = T-1, use all of current frame and none from
                    # history
                    with tf.variable_scope(attention_input_scope or 'attn_time_input',
                                           reuse=t!=0) as attention_input_scope:
                        attention_input = tf.concat([history[:, t + 1:, :],
                                                    x[:, : t + 1, :]],
                                                    1, name='window')

                elif self.attention_layer and self.attention_input == 'image':
                    with tf.variable_scope(attention_input_scope or 'attn_img_input',
                                           reuse=t!=0) as attention_input_scope:
                        D_enc = np.prod(x.shape.as_list()[2:])
                        D_patch = tf.cast(D_enc / self.L, tf.int32)
                        attention_input = tf.reshape(x[:, t, :], [-1, self.L, D_patch])
                elif self.attention_layer:
                    raise ValueError('Unknown attention_input type {}'.format(self.attention_input))

                if self.attention_layer:
                    with tf.variable_scope(attention_scope or 'attention', reuse=t != 0) as attention_scope:
                        attention, alpha = do_attention(attention_input, c, t != 0)
                        attentions.append(attention)
                        alphas.append(alpha)

                        if self.enc2ctx:
                            flat_history = tf.reshape(
                                x, [batch_size, -1],
                                name='flat_history')
                            context = tf.concat(
                                [flat_history, attention],
                                1, name='enc2ctx')
                        else:
                            context = attention

                        # TODO checkout where to best put dropout with attention
                        if use_dropout:
                            context = tf.nn.dropout(context, self.dropout)
                else:
                    context = x[:, t, :]

                with tf.name_scope(decode_lstm_scope or 'final_decode') as decode_lstm_scope:
                    output = self._decode(context, h, is_training, reuse=(t != 0))
                    outputs.append(output)

            outputs = tf.stack(outputs, axis=1)  # B x T x 1
            if self.attention_layer:
                attentions = tf.stack(attentions, axis=1, name='attention')
                alphas = tf.stack(alphas, axis=1, name='alpha')

        control_deps = []

        # Saves LSTM state so decoding can continue like normal in a next call
        if state_saver is not None:
            save_state_scope = 'save_state'
            with tf.variable_scope(save_state_scope):
                state_saves = [
                    state_saver.save_state('lstm_c', c, 'lstm_c'),
                    state_saver.save_state('lstm_h', h, 'lstm_h'),
                    state_saver.save_state('first', tf.zeros(
                        [batch_size], dtype=tf.bool), 'first'),
                    state_saver.save_state('history', x, 'history'),
                    state_saver.save_state('context', context, 'context'),
                    state_saver.save_state('output', output, 'last_output'),
                    state_saver.save_state('target', targets[:,-1,:], 'output')
                ]
                save_state = tf.group(*state_saves)
                control_deps.append(save_state)

                # Makes it easier by just injecting the save state control op
                # into the rest of the computation graph, but also makes it
                # messy
                with tf.control_dependencies(control_deps):
                    lengths = state_saver.length
                    # Tricksy way of injecting dependency
                    outputs = tf.identity(outputs, name='output')
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
            'output': tf.identity(outputs, name='output'),
        }
        tf_util.add_to_collection(attend.GraphKeys.OUTPUT, out['output'])

        if self.attention_layer:
            out.update({
                'attention': tf.identity(attentions, name='attention'),
                'alpha': tf.identity(alphas, name='alpha'),
            })
            tf_util.add_to_collection(
                attend.GraphKeys.OUTPUT, [
                    out['attention'], out['alpha']])

        return out, context

    def _extract_keys(self, keys):
        with tf.name_scope('extract_keys'):
            # Assume key looks like 'seqn:original_key:random', so 3 splits
            splits = tf.string_split(keys, ':')
            splits = tf.reshape(splits.values, ([-1, 3]))  # Reshape per key
            splits = splits[:, 1:]  # Drop the sequence info
            keys = tf.map_fn(
                lambda x: tf.string_join(
                    tf.unstack(x), ':'), splits)
            keys = tf.identity(keys, name='key')
            return keys

    def calculate_loss(self, predictions, targets, lengths=None):
        # predictions.shape.assert_is_compatible_with(targets)

        with tf.variable_scope('loss'):
            T = tf.shape(targets)[1]
            # targets = tf.squeeze(targets) # Get rid of trailing 1-dimensions
            # Tails of sequences are likely padded, so create a mask to ignore
            # padding
            mask = tf.cast(
                tf.sequence_mask(lengths, T),
                tf.int32) if lengths is not None else None
            if predictions.shape.ndims > 2:
                mask = tf.expand_dims(mask, -1)

            loss = self.loss_fun(targets, predictions, weights=mask)

        # if self.debug:
        #     loss = tf.Print(loss, [loss], message='loss ')

        return loss

    # It's important to have losses per entire sequence
    def calculate_losses(
            self,
            predictions,
            targets,
            keys,
            lengths=None,
            scope='loss'):
        from attend import metrics

        with tf.variable_scope(scope):
            if targets.shape.ndims > 2:
                targets = tf.squeeze(targets, axis=2, name='squeeze_target')
            if predictions.shape.ndims > 2:
                predictions = tf.squeeze(
                    predictions, axis=2, name='squeeze_pred')
            shape = tf.shape(predictions)
            # B, T = shape[0], shape[1]
            T = shape[1]
            T = self.T
            length_table = tf.contrib.lookup.MutableHashTable(
                key_dtype=tf.string, value_dtype=tf.int64, default_value=0, name='length_table')

            mask = tf.cast(tf.sequence_mask(lengths, T), tf.float32)
            # if predictions.shape.ndims > 2:
            #     mask = tf.expand_dims(mask, -1)
            length_update = length_table.insert(
                keys, tf.cast(lengths, tf.int64))

            out = dict(batch={}, all={}, total={})

            # loss_tables = {}
            # batch_losses = {}

            # batch_mse, mse_table = metrics.streaming_mse(predictions, targets,
            #                                              keys, lengths, mask)
            # batch_losses['mse'] = batch_mse
            # loss_tables['mse'] = mse_table

            # Only use the safe, non-racey streaming output
            _, corr = tf.contrib.metrics.streaming_pearson_correlation(
                predictions, targets, mask)
            _, mse = tf.contrib.metrics.streaming_mean_squared_error(
                predictions, targets, mask)
            # ICC needs fully defined shapes
            B_max = self.provider.batch_size
            B_actual = shape[0]
            padding_shape = [B_max - B_actual, *tf.unstack(shape)[1:]]
            # padding = tf.zeros([predictions., ])
            padding = [[0, B_max-B_actual]] + [[0,0]] * (predictions.shape.ndims - 1)
            # padding = tf.co
            # padded_predictions = tf.concat([predictions, tf.zeros(padding_shape)], axis=0)
            padded_predictions = tf.pad(predictions, padding)
            padded_targets = tf.pad(targets, padding)
            padded_mask = tf.pad(mask, padding)
            final_shape = [B_max] + predictions.shape.as_list()[1:]
            padded_predictions.set_shape(final_shape)
            padded_targets.set_shape(final_shape)
            padded_mask.set_shape(final_shape)
            icc = attend.metrics.streaming_icc(3,1)(
                padded_predictions, padded_targets, padded_mask)

            streaming_vars = tf.contrib.framework.get_local_variables(tf.get_variable_scope())
            streaming_reset = tf.variables_initializer(streaming_vars)

            with tf.control_dependencies([length_update]):
                all_keys, all_lengths = length_table.export()
                out['context'] = {'all_keys': all_keys,
                                  'all_lengths': all_lengths}
                # for k in batch_losses.keys():
                #     out['batch'][k] = batch_losses[k]
                #     out['all'][k] = loss_tables[k].export()[1]

                out['total']['pearson_r'] = corr
                out['total']['mse'] = mse # Used to be mse_tf
                out['total']['icc'] = icc

            return out, streaming_reset

    def _decode(self, x, h, is_training, reuse=False):
        with tf.variable_scope('decode', reuse=reuse):
            decode_input = tf.concat([x, h], 1, name='decode_input')
            out = tf.layers.dense(decode_input, 1, activation=None,
                                  kernel_initializer=self.weight_initializer,
                                  name='final_dense')

            if self.use_batch_norm:
                # TODO renorm_clipping: {'rmax', 'rmin', 'dmax'}
                # In the original paper they anneal those values
                out = tf.contrib.layers.batch_norm(out, decay=self.batch_norm_decay,
                                                   center=True, scale=True,
                                                   is_training=is_training,
                                                   renorm=self.use_batch_renorm)

            if self.final_activation:
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

            w_h = tf.get_variable(
                'w_h', [D, self.H],
                initializer=self.weight_initializer)
            b_h = tf.get_variable(
                'b_h', [self.H],
                initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable(
                'w_c', [D, self.H],
                initializer=self.weight_initializer)
            b_c = tf.get_variable(
                'b_c', [self.H],
                initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
        return c, h
