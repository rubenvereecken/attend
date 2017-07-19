import tensorflow as tf
import numpy as np
from time import time

from util import *
from attend.log import Log; log = Log.get_logger(__name__)

class AttendSolver():
    def __init__(self, model, update_rule, learning_rate):
        self.model = model

        self.update_rule = update_rule
        self.learning_rate = learning_rate

        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        else:
            raise Exception()


    def train(self, num_epochs, steps_per_epoch, batch_size, time_steps,
              provider, # TODO not really used?
              encoder,
              log_dir,
              debug=False):

        # For now, its results are stored inside the provider
        provider.batch_sequences_with_states()
        # provider.batch_static_pad()

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
            # for v in tf.trainable_variables():
            #     print(v)
            #     print(v.shape)
            num_vars = np.sum(list(map(lambda v: np.prod(v.shape), tf.trainable_variables())))
            log.info('Total trainable vars {}'.format(num_vars))

        # Initialize variables
        # The Supervisor actually runs this automatically, but I'm keeping this
        init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

        # Summary op
        tf.summary.scalar('batch_loss', loss_op)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        # for grad, var in grads_and_vars:
        #     tf.summary.histogram(var.op.name+'/gradient', grad)

        # The Supervisor already merges all summaries but I like explicit
        summary_op = tf.summary.merge_all()

        # The Supervisor saves summaries after X seconds, not good for model progressions
        # sv = tf.train.Supervisor(logdir=log_dir, summary_op=summary_op,
        #         save_summaries_secs=0)
        # coord = sv.coord
        coord = tf.train.Coordinator()
        if debug:
            config = tf.ConfigProto(
                # intra_op_parallelism_threads=1
            )
        else:
            config = tf.ConfigProto()

        # Managed session will do the necessary init_ops, start queue runners,
        # start checkpointing/summary service
        # It will also recover from a checkpoint if available
        # with sv.managed_session(config=config) as sess:
        with tf.Session() as sess:
            sess.run(init_op)
            # Special input runners run separately because the supervisor can't
            # serialize them
            # input_threads = tf.train.start_queue_runners(sess=sess, coord=coord,
            #         collection='input_runners')
            input_threads = []
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

            t_start = time()
            log.debug('Started')

            try:
                # while not sv.should_stop():
                while not coord.should_stop():
                    # Run batch_loss summary op together with loss_op
                    # Otherwise it will recompute the loss separately
                    loss, _, summary = sess.run([loss_op, train_op, summary_op])
                    global_step_value = tf.train.global_step(sess, global_step)

                    summary_writer.add_summary(summary, global_step_value)
                    # import pdb; pdb.set_trace()

                    # Run training steps or whatever
                    if global_step_value % steps_per_epoch == 0:
                        # TODO do some post epoch stuff
                        pass
                    if global_step_value == steps_per_epoch * num_epochs:
                        # TODO you done
                        # sv.request_stop()
                        log.debug('Completed final step')
                        coord.request_stop()
            except tf.errors.OutOfRangeError:
                log.info('Done training -- epoch limit reached')
                notify('Done training', 'Took {:.1f}s'.format(time() - t_start))
            except Exception as e:
                log.critical(e)
                notify('Error occurred', 'Took {:.1f}s'.format(time() - t_start))
            finally:
                log.debug('Finally - ...')
                # Requests the coordinator to stop, joins threads
                # and closes the summary writer if enabled through supervisor
                coord.join(threads + input_threads)
                # sv.stop()
                coord.stop()

            sess.close()
