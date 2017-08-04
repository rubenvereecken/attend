import tensorflow as tf
import numpy as np
import cson
import os

from attend import util
from attend import Encoder, AttendSolver, AttendModel
from attend.provider import Provider, InMemoryProvider

class Evaluator:
    def __init__(self, encoder, provider, model, args=None, scope=None):
        self.encoder = encoder
        self.provider = provider
        self.model = model
        self.args = args
        self.scope = scope or ''

        self.graph = tf.Graph()
        self.out_ops, self.ctx_ops = self._build_model()
        self.loss_ops = self._build_losses()

        self.sess = tf.Session(graph=self.graph)
        # from tensorflow.python import debug as tf_debug
        # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

        self.sess.run(self._variables_initializer('losses'))

        with self.graph.as_default():
            self.reset_op = self.provider.state_saver.reset_states()


    def _build_model(self):
        with self.graph.as_default():
            with tf.variable_scope(self.scope):
                self.provider.batch_sequences_with_states()
                out_ops, ctx_ops = self.model.build_model(self.provider, False)
        return out_ops, ctx_ops


    def _build_losses(self):
        with self.graph.as_default():
            # with tf.variable_scope(self.scope):
            loss_ops = self.model.calculate_losses(self.out_ops['output'], self.provider.targets,
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


    # def initialize_variables(self):
    #     self.init_op = tf.group(tf.global_variables_initializer(),
    #                             tf.local_variables_initializer())
    #     self.sess.run(self.init_op)

    @property
    def variables(self):
        return self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

#     def __del__(self):
#         print('Cleaning up session')
#         self.sess.close()


    @classmethod
    def init_from_logs(cls, log_dir, feat_dim=(272,)):
        """
        Rebuilds the model from logs based on the saved cli arguments
        and saved network weights.
        There can be a discrepancy between the code at that time and now.
        """
        if not os.path.exists(log_dir + '/args.cson'):
            raise Exception('No args.cson file found in {}'.format(log_dir))

        checkpoint = tf.train.latest_checkpoint(log_dir)
        if checkpoint is None:
            raise Exception('No checkpoints file found in {}'.format(log_dir))

        with open(log_dir + '/args.cson', 'r') as f:
            args = cson.load(f)

        encoder = util.init_with(Encoder, args)
        provider = InMemoryProvider(encoder=encoder, dim_feature=feat_dim,
                **util.pick(args, util.params_for(Provider.__init__)))
        model = AttendModel(provider, encoder,
                **util.pick(args, util.params_for(AttendModel.__init__)))

        evaluator = cls(encoder, provider, model, args)
        saver = tf.train.Saver(evaluator.variables)
        saver.restore(evaluator.sess, checkpoint)
        return evaluator


    def evaluate(self, sequence, targets=None, key='input_seq'):
        """
        In memory evaluation
        """

        T = self.model.T
        D = 256
        l = sequence.shape[0]
        n_batches = np.ceil(l / T).astype(int)

        total = { 'output': np.empty((l,1)) }

        if 'context' in self.out_ops:
            total.update({
                  'context': np.empty((l, D)),
                  'alpha': np.empty((l, D)),
                })

        state_saver = self.provider.state_saver
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
                total[k][offset:offset+batch_l] = v[0][:batch_l]
            keys.append(ctx['key'][0].decode())

        # Reset saved states for this sequence
        keys = list(set(keys))
        self.sess.run(self.reset_op, { state_saver._key: keys })

        if not targets is None:
            total.update(total_loss)

        return total
