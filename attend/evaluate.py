import tensorflow as tf
import numpy as np
import cson
import os

import attend
from attend import util, tf_util
from attend import Encoder, AttendSolver, AttendModel
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


    def initialize(self):
        self.out_ops, self.ctx_ops = self._build_model()
        self.loss_ops = self._build_losses()
        self.reset_op = self._build_reset()

        self._initialize_variables()


    def _initialize_variables(self):
        with self.graph.as_default():
            # checkpoint = self.log_dir + '/model.ckpt-50000'
            checkpoint = tf.train.latest_checkpoint(self.log_dir)
            if checkpoint is None:
                tf.logging.warning('No checkpoint file found in {}; reinitializing'.format(self.log_dir))

            if not checkpoint is None:
                self.sess.run(self._variables_initializer('losses'))
                saver = self.get_saver()
                saver.restore(self.sess, checkpoint)
            else:
                self.sess.run(self._variables_initializer())


    def _build_model(self):
        raise NotImplementedError()


    def _build_losses(self):
        with self.graph.as_default():
            # with tf.variable_scope(self.scope):
            loss_ops = self.model.calculate_losses(self.out_ops['output'],
                    self.state_saver._sequences['conflict'],
                    self.ctx_ops['key'], self.ctx_ops['length'], 'losses')
        return loss_ops


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
        self.sess.run(self.reset_op, { self.state_saver._key: keys })


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


    def evaluate(self, sequence, targets=None, key='input_seq'):
        """
        In memory evaluation
        """

        T = self.model.T
        D = 256
        l = sequence.shape[0]
        n_batches = np.ceil(l / T).astype(int)

        # total = { 'output': np.empty((l,1)) }
        total = dict(output=[])

        if 'alpha' in self.out_ops:
            total.update(dict(attention=[], alpha=[]))

        state_saver = self.state_saver
        key = '{}:{}'.format(key, '0')
        keys = []

        # history = np.zeros(())

        for i in range(n_batches):
            offset = i * T
            batch = sequence[offset:offset+T]
            batch_l = batch.shape[0]
            batch = util.pad(batch, T-batch_l)

            feed_dict = { state_saver._sequences['images']: [batch],
                        state_saver._full_key: ['{:05d}_of_{:05d}:{}'.format(i,
                            n_batches, key)],
                        state_saver._key: [key],
                        state_saver._length: [batch_l],
                        state_saver._sequence: [i],
                        state_saver._sequence_count: [n_batches],
                        }

            ctx_ops = self.ctx_ops.copy()

            if not targets is None:
                target_batch = targets[offset:offset+T]
                target_batch = util.pad(target_batch, T-batch_l)
                feed_dict[state_saver._sequences['conflict']] = [target_batch]

                ctx_ops.update(self.loss_ops['context'])
                out, ctx, total_loss = self.sess.run((self.out_ops, ctx_ops, self.loss_ops['total']), feed_dict=feed_dict)

            else:
                out, ctx = self.sess.run((self.out_ops, ctx_ops), feed_dict=feed_dict)

            for k, v in out.items():
                total[k].append(v[0][:batch_l])
                # total[k][offset:offset+batch_l] = v[0][:batch_l]
            keys.append(ctx['key'][0].decode())

        # Reset saved states for this sequence
        keys = list(set(keys))
        self.reset(keys)

        total = { k: np.concatenate(v) for k, v in total.items() }

        if not targets is None:
            total.update(total_loss)

        total['output'] = np.resize(total['output'], [l])

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
    def import_from_logs(cls, log_dir, feat_dim=(272,), meta_path='eval_model.meta.proto'):
        if not os.path.isdir(log_dir):
            raise Exception('Log dir does not exist')

        args = Evaluator.get_args(log_dir)

        encoder = util.init_with(Encoder, args)
        provider = InMemoryProvider(encoder=encoder, dim_feature=feat_dim,
                **util.pick(args, util.params_for(Provider.__init__)))
        model = AttendModel(provider, encoder,
                **util.pick(args, util.params_for(AttendModel.__init__)))

        e = cls(model, log_dir, meta_path, args)
        e.initialize()

        return e


class RebuildEvaluator(Evaluator):
    def __init__(self, encoder, provider, model, log_dir, args=None, scope=None):
        self.encoder = encoder
        self.provider = provider
        self.model = model

        super().__init__(log_dir, args, scope)


    def _build_model(self):
        with self.graph.as_default():
            with tf.variable_scope(self.scope):
                self.provider.batch_sequences_with_states()
                # State saver gets created during batching
                self.state_saver = self.provider.state_saver
                out_ops, ctx_ops = self.model.build_model(self.provider, False)
        return out_ops, ctx_ops


    def get_saver(self):
        saver = tf.train.Saver(self.variables)
        return saver


    def _build_reset(self):
        with self.graph.as_default():
            reset_op = self.provider.state_saver.reset_states()
            return reset_op


    @classmethod
    def rebuild_from_logs(cls, log_dir, extra_args={}, feat_dim=(272,)):
        """
        Rebuilds the model from logs based on the saved cli arguments
        and saved network weights.
        There can be a discrepancy between the code at that time and now.
        """
        if not os.path.isdir(log_dir):
            raise Exception('Log dir does not exist')
        args = Evaluator.get_args(log_dir)
        args.update(extra_args)

        encoder = util.init_with(Encoder, args)
        provider = InMemoryProvider(encoder=encoder, dim_feature=feat_dim,
                **util.pick(args, util.params_for(Provider.__init__)))
        model = AttendModel(provider, encoder,
                **util.pick(args, util.params_for(AttendModel.__init__)))

        evaluator = cls(encoder, provider, model, log_dir, args)
        evaluator.initialize()

        return evaluator


class DictProxy(object):
    def __init__(self, d):
        object.__setattr__(self, '_d', d)


    def __getattribute__(self, name):
        return object.__getattribute__(self, '_d').get(name)
