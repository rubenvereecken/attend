import tensorflow as tf
import numpy as np

from util import *

class AttendSolver():
    def __init__(self, model, update_rule, learning_rate):
        self.model = model

        self.update_rule = update_rule
        self.learning_rate = learning_rate

        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        else:
            raise Exception()


    def train(self, num_epochs, batch_size, time_steps,
              provider, # TODO not really used?
              encoder,
              log_dir,
              debug=False):

        # For now, its results are stored inside the provider
        provider.batch_sequences_with_states()

        global_step = tf.Variable(0, trainable=False, name='global_step')
        loss_op = self.model.build_model(provider)

        with tf.name_scope('optimizer'):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            # train_op = optimizer.minimize(loss_op, global_step=global_step,
            #         var_list=tf.trainable_variables())
            grads = tf.gradients(loss_op, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars,
                    global_step=global_step)
            for v in tf.trainable_variables():
                print(v)
                print(v.shape)
            num_vars = np.sum(list(map(lambda v: np.prod(v.shape), tf.trainable_variables())))
            print('Total trainable vars {}'.format(num_vars))

        # Initialize variables
        # The Supervisor actually runs this automatically, but I'm keeping this
        init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

        # TODO one of these silly summary ops makes the model run twice the first time
        # Summary op
        tf.summary.scalar('batch_loss', loss_op)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        # for grad, var in grads_and_vars:
        #     tf.summary.histogram(var.op.name+'/gradient', grad)

        # The Supervisor already merges all summaries but I like explicit
        summary_op = tf.summary.merge_all()

        # The Supervisor saves summaries after X seconds, not good for model progressions
        sv = tf.train.Supervisor(logdir=log_dir, summary_op=summary_op,
                save_summaries_secs=0)
        if debug:
            config = tf.ConfigProto(
                intra_op_parallelism_threads=4
            )
        else:
            config = tf.ConfigProto()

        # Managed session will do the necessary init_ops, start queue runners,
        # start checkpointing/summary service
        # It will also recover from a checkpoint if available
        with sv.managed_session(config=config) as sess:
            summary_writer = tf.summary.FileWriter(log_dir)

            try:
                while not sv.should_stop():
                    loss, _ = sess.run([loss_op, train_op])
                    global_step_value = tf.train.global_step(sess, global_step)
                    summary = sess.run(summary_op)
                    summary_writer.add_summary(summary, global_step_value)
                    break
                    # Run training steps or whatever
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            except Exception as e:
                print(e)
            finally:
                # Requests the coordinator to stop, joins threads
                # and closes the summary writer if enabled through supervisor
                sv.stop()

            sess.close()
