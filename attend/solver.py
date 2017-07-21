import tensorflow as tf
import numpy as np
from time import time


from util import *
from attend.log import Log; log = Log.get_logger(__name__)
import attend

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
              provider,
              encoder,
              log_dir,
              val_provider=None,
              debug=False,
              show_progress_bar=None):

        if show_progress_bar is None and debug is False:
            show_progress_bar = True
        if show_progress_bar:
            from tqdm import tqdm
            progress_wrapper = tqdm
        else:
            progress_wrapper = lambda i, **kwargs: i

        g = tf.get_default_graph()

        # For now, its results are stored inside the provider
        provider.batch_sequences_with_states()
        # provider.batch_static_pad()

        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Prediction and loss
        with tf.variable_scope(tf.get_variable_scope()):
            outputs, lengths = self.model.build_model(provider, train=True)
            loss_op = self.model.calculate_loss(outputs, provider.targets, lengths)

        if not val_provider is None:
            val_provider.batch_sequences_with_states(1,
                    collection=attend.GraphKeys.VAL_INPUT_RUNNERS,
                    container_name='inputcontainer')
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                # tf.get_variable_scope().reuse_variables()
                val_outputs, val_lengths = self.model.build_model(val_provider, train=False)
                val_losses = self.model.calculate_losses(val_outputs,
                        val_provider.targets, val_lengths, 'val_loss')

        with tf.variable_scope('optimizer', reuse=False):
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
        with tf.Session(graph=g) as sess:
            sess.run(init_op)
            # Special input runners run separately because the supervisor can't
            # serialize them
            # input_threads = tf.train.start_queue_runners(sess=sess, coord=coord,
            #         collection='input_runners')
            input_threads = []
            runners = g.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
            log.debug('%s', [r.name for r in runners])
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

            t_start = time()
            log.info('Started training')

            try:
                # while not coord.should_stop():
                for epoch_i in progress_wrapper(range(num_epochs)):
                    for step_i in progress_wrapper(range(steps_per_epoch)):
                        # Run batch_loss summary op together with loss_op
                        # Otherwise it will recompute the loss separately
                        loss, _, summary = sess.run([loss_op, train_op, summary_op])
                        global_step_value = tf.train.global_step(sess, global_step)
                        log.debug('TRAIN %s - %s', global_step_value, loss)

                        summary_writer.add_summary(summary, global_step_value)

                        if coord.should_stop():
                            break
                        # END STEP

                    if coord.should_stop():
                        break

                    # Validation after every epoch
                    if val_provider:
                        # Fire up the validation input fetching threads first time
                        if epoch_i == 0:
                            val_input_threads = tf.train.start_queue_runners(sess=sess,
                                    coord=coord, collection=attend.GraphKeys.VAL_INPUT_RUNNERS)
                            threads += val_input_threads

                        try:
                            for i in range(1000):
                                losses = sess.run([val_losses])
                                log.debug('TEST  %s, %s', i, losses)
                        except tf.errors.OutOfRangeError:
                            log.info('Finished validation')

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
                # coord.stop() DOESNT EXIST

            sess.close()
