import tensorflow as tf
import numpy as np
import cson
import os

import attend
from attend import util, tf_util
from attend import Encoder, AttendSolver
from attend.model import AttendModel
from attend.provider import Provider, InMemoryProvider


class Evaluator:
    def __init__(self, log_dir, args=None, scope=None):
        self.log_dir = log_dir
        self.args = args
        self.scope = scope or ''

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        # from tensorflow.python import debug as tf_debug
        # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        self.build_inits()


    def build_inits(self):
        self.out_ops, self.ctx_ops = self._build_model()
        self.loss_ops, self.loss_reset_op = self._build_losses()
        self.reset_op = self._build_reset()
        self.loss_init = self._variables_initializer('losses')
        self.var_init = self._variables_initializer()


    def load_checkpoint(self, ckpt=None):
        with self.graph.as_default():
            if ckpt is None:
                checkpoint = tf.train.latest_checkpoint(self.log_dir)
            elif isinstance(ckpt, int):
                state = tf.train.get_checkpoint_state(self.log_dir)
                try:
                    checkpoint = next(p for p in state.all_model_checkpoint_paths \
                                    if p.endswith(str(ckpt)))
                except:
                    raise Exception('Checkpoint {} not found'.format(ckpt))
                checkpoint_path = state

            if not checkpoint is None:
                self.sess.run(self.loss_init)
                saver = self.get_saver()
                saver.restore(self.sess, checkpoint)
            else:
                tf.logging.warning('No checkpoint file found in {}; reinitializing'.format(self.log_dir))
                self.sess.run(self.var_init)


    def _build_model(self):
        raise NotImplementedError()


    def _build_losses(self):
        with self.graph.as_default():
            loss_ops, loss_reset = self.model.calculate_losses(self.out_ops['output'],
                    self.state_saver._sequences['conflict'],
                    self.ctx_ops['key'], self.ctx_ops['length'], 'losses')
            # TODO for icc
            self.placeholders_for_loss = dict(
                predictions=tf.placeholder(tf.float32, [None, None, 1]),
                targets=tf.placeholder(tf.float32, [None, None, 1]),
                lengths=tf.placeholder(tf.int32, [None])
            )
            from attend.losses import icc
            self.icc_loss = self.model.calculate_loss(
                self.placeholders_for_loss['predictions'],
                self.placeholders_for_loss['targets'],
                self.placeholders_for_loss['lengths'], icc(3,1))
        return loss_ops, loss_reset


    def write_graph(self, log_dir):
        import os
        os.makedirs(log_dir, exist_ok=True)
        writer = tf.summary.FileWriter(log_dir, graph=self.graph)


    def _variables_initializer(self, scope=None):
        with self.graph.as_default():
            init_op = tf.variables_initializer(
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope) + \
                    tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=scope))

        return init_op


    def reset(self, keys):
        """
        Reset state variables and streaming metrics
        """
        self.sess.run([self.reset_op, self.loss_reset_op], { self.state_saver._key: keys })


    def run(self, *args, **kwargs):
        return self.sess.run(*args, **kwargs)


    def get_collection_as_dict(self, key):
        return tf_util.get_collection_as_dict(key, self.graph)


    @property
    def variables(self):
        return self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)


    @classmethod
    def get_args(cls, log_dir):
        if not os.path.exists(log_dir + '/args.cson'):
            raise Exception('No args.cson file found in {}'.format(log_dir))

        with open(log_dir + '/args.cson', 'r') as f:
            args = cson.load(f)

        return args


    def evaluate(self, sequence, targets=None, keys=None, feed_dict={}):
        """
        In memory evaluation
        """

        if isinstance(sequence, list):
            # Assume multiple, pad to one
            lengths = [len(seq) for seq in sequence]
            from attend.util import pad_and_stack
            sequences = pad_and_stack(sequence)
        else:
            lengths = [len(sequence)]
            sequences = np.expand_dims(sequence, 0)

        if isinstance(targets, list):
            targets = pad_and_stack(targets)
        else:
            targets = np.expand_dims(targets, 0)

        del sequence

        T = self.model.T
        l = sequences.shape[1]
        n_batches = np.ceil(l / T).astype(int)
        batch_size = sequences.shape[0]

        # total = { 'output': np.empty((l,1)) }
        total = dict(output=[])

        if 'alpha' in self.out_ops:
            total.update(dict(attention=[], alpha=[]))

        state_saver = self.state_saver
        if keys is None:
            keys = ['input_seq:{}'.format(i) for i in range(batch_size)]
        else:
            keys = ['{}:{}'.format(k, i) for i, k in enumerate(keys)]
        out_keys = []

        # history = np.zeros(())

        import tqdm
        for i in tqdm.tqdm(range(n_batches)):
            offset = i * T
            batch = sequences[:,offset:offset+T]
            lengths_used = np.ones_like(lengths) * offset
            leftover_lengths = np.max(np.array([np.zeros_like(lengths),
                                                lengths - lengths_used]), axis=0)
            batch_lengths = np.min(np.array([np.ones_like(lengths) * T,
                                             leftover_lengths]), axis=0)
            batch_l = T
            if batch.shape[1] != T:
                batch = util.pad(batch, T-batch.shape[1], 1)

            state_dict = { state_saver._sequences['images']: batch,
                        state_saver._full_key: ['{:05d}_of_{:05d}:{}'.format(i,
                            n_batches, key) for key in keys],
                        state_saver._key: keys,
                        state_saver._length: batch_lengths,
                        state_saver._sequence: [i] * batch_size,
                        state_saver._sequence_count: [n_batches] * batch_size,
                        }
            feed_dict = feed_dict.copy()
            feed_dict.update(state_dict)

            ctx_ops = self.ctx_ops.copy()

            if not targets is None:
                target_batch = targets[:,offset:offset+T]
                if target_batch.shape[1] != T:
                    target_batch = util.pad(target_batch, T-target_batch.shape[1], 1)
                # target_batch = util.pad(target_batch, T-batch_l)
                feed_dict[state_saver._sequences['conflict']] = target_batch

                ctx_ops.update(self.loss_ops['context'])
                out, ctx, total_loss = self.sess.run((self.out_ops, ctx_ops, self.loss_ops['total']), feed_dict=feed_dict)

            else:
                out, ctx = self.sess.run((self.out_ops, ctx_ops), feed_dict=feed_dict)

            for k, v in out.items():
                total[k].append(v)
                # total[k][offset:offset+batch_l] = v[0][:batch_l]
            out_keys.extend([key.decode() for key in ctx['key']])

        # Reset saved states for this sequence
        out_keys = list(set(out_keys))
        self.reset(out_keys)

        total = { k: np.concatenate(v, 1) for k, v in total.items() }

        # Calculate ICC
        assert not targets is None, 'for now'
        icc_score = self.sess.run(self.icc_loss, {
            self.placeholders_for_loss['predictions']: (total['output'][:,:max(lengths)]),
            self.placeholders_for_loss['targets']: np.expand_dims(targets, -1),
            self.placeholders_for_loss['lengths']: lengths,
        })
        total['icc'] = icc_score

        total['output'] = util.unstack_and_unpad(total['output'], lengths)

        if len(total['output']) == 1:
            total['output'] = total['output'][0]

        if not targets is None:
            total.update(total_loss)


        # total['output'] = np.resize(total['output'], [l])

        return total


class ImportEvaluator(Evaluator):
    def __init__(self, model, log_dir, meta_path, args=None, scope=None):
        self.model = model
        self.meta_path = meta_path

        super().__init__(log_dir, args, scope)


    def _build_model(self):
        with self.graph.as_default():
            # Imports the graph as well
            try:
                self.saver = tf.train.import_meta_graph(self.log_dir + '/' + self.meta_path)
            except OSError as e:
                raise Exception('eval_model.meta.proto not found, maybe an old implementation?')

            input = tf_util.get_collection_as_dict(attend.GraphKeys.INPUT)
            state = self.graph.get_collection(attend.GraphKeys.STATE_SAVER)
            state = { '_' + tf_util.name(v): v for v in state }
            state['_sequences'] = {
                'images': input['features'],
                'conflict': input['targets'],
            }
            self.state_saver = DictProxy(state)
            output = tf_util.get_collection_as_dict(attend.GraphKeys.OUTPUT)
            context = tf_util.get_collection_as_dict(attend.GraphKeys.CONTEXT)

            return output, context


    def get_saver(self):
        return self.saver


    def _build_reset(self):
        reset_op = self.graph.get_collection(attend.GraphKeys.STATE_RESET)[0]
        return reset_op


    @classmethod
    def import_from_logs(cls, log_dir, feat_dim=(272,), meta_path='eval_model.meta.proto',
                         initialize=True):
        if not os.path.isdir(log_dir):
            raise Exception('Log dir does not exist')

        args = Evaluator.get_args(log_dir)

        encoder = util.init_with(Encoder, args)
        provider = InMemoryProvider(encoder=encoder, dim_feature=feat_dim,
                **util.pick(args, util.params_for(Provider.__init__)))
        model = AttendModel(provider, encoder,
                **util.pick(args, util.params_for(AttendModel.__init__)))

        e = cls(model, log_dir, meta_path, args)
        if initialize:
            e.load_checkpoint()

        return e


class RebuildEvaluator(Evaluator):
    def __init__(self, encoder, provider, model, log_dir, args=None, scope=None, is_training=False):
        self.encoder = encoder
        self.provider = provider
        self.model = model
        self.is_training = is_training

        super().__init__(log_dir, args, scope)


    def _build_model(self):
        with self.graph.as_default():
            with tf.variable_scope(self.scope):
                self.provider.batch_sequences_with_states()
                # State saver gets created during batching
                self.state_saver = self.provider.state_saver
                out_ops, ctx_ops = self.model.build_model(self.provider,
                                                          self.is_training)
        return out_ops, ctx_ops


    def get_saver(self):
        saver = tf.train.Saver(self.variables)
        return saver


    def _build_reset(self):
        with self.graph.as_default():
            reset_op = self.provider.state_saver.reset_states()
            return reset_op


    def evaluate(self, sequence, targets=None, key=None, feed_dict={},
                 epsilon=None):
        """
        Arguments:
            epsilon: to be added to feed_dict; convenience argument
        """
        if self.is_training and targets is None:
            raise ValueError('Targets should be provided in training mode (for the sampler)')

        feed_dict = feed_dict.copy()

        if not epsilon is None:
            epsilon_tensor = tf_util.get_collection_as_singleton(
                attend.GraphKeys.SAMPLING_EPSILON, self.graph
            )

            feed_dict.update({ epsilon_tensor: epsilon })

        return super().evaluate(sequence, targets, key, feed_dict=feed_dict)


    @classmethod
    def rebuild_from_logs(cls, log_dir, extra_args={},
                          feat_dim=(272,), is_training=False,
                          initialize=True):
        """
        Rebuilds the model from logs based on the saved cli arguments
        and saved network weights.
        There can be a discrepancy between the code at that time and now.
        """
        if not os.path.isdir(log_dir):
            raise Exception('Log dir does not exist')
        args = Evaluator.get_args(log_dir)
        args.update(extra_args)

        if is_training and args['sampling_scheme'] not in ['standard', None] \
                and args.get('sampler_kwargs', {}).get('epsilon', None) is None:
            sampler_kwargs = args.get('sampler_kwargs', {})
            sampler_kwargs['epsilon'] = 0
            args['sampler_kwargs'] = sampler_kwargs
            tf.logging.warn('Falling back to sampling epsilon 0, can still be overridden with feed_dict')

        if is_training and args['sampling_scheme'] not in ['standard', None]:
            args['sampling_scheme'] = 'fixed_epsilon'
            tf.logging.info('Changed sampling scheme to fixed_epsilon')

        encoder = util.init_with(Encoder, args)
        provider = InMemoryProvider(encoder=encoder, dim_feature=feat_dim,
                **util.pick(args, util.params_for(Provider.__init__)))
        model = AttendModel(provider, encoder,
                **util.pick(args, util.params_for(AttendModel.__init__)))

        evaluator = cls(encoder, provider, model, log_dir, args,
                        is_training=is_training)
        if initialize:
            evaluator.load_checkpoint()

        return evaluator


class DictProxy(object):
    def __init__(self, d):
        object.__setattr__(self, '_d', d)


    def __getattribute__(self, name):
        return object.__getattribute__(self, '_d').get(name)
