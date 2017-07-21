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

    def test(self, graph, saver, save_path, provider, context_ops, loss_ops,
            summary_writer=None):
        # Reasons for building a new session every validation step:
        # - There is no way to keep track of epochs OR to restart queues
        #   so a new session keeps it easy to loop through the input
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess = tf.Session(graph=graph)
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,
                coord=coord, collection=attend.GraphKeys.VAL_INPUT_RUNNERS)

        # Load trained variables into the shared graph inside the val session
        saver.restore(sess, save_path)

        # These are batch losses per key
        # Might be interesting to see errors progress over time within a vid
        losses_by_key = {}

        try:
            for i in range(1000000000):
                fetch = {}
                fetch.update(context_ops)
                fetch.update(loss_ops)
                out = sess.run(fetch)
                original_keys = list(map(lambda k: str(k).split(':')[-1], out['key']))
                for i, key in enumerate(original_keys):
                    idx = out['sequence_idx'][i]
                    if key not in losses_by_key:
                        count = out['sequence_count'][i]
                        losses_by_key[key] = np.zeros(count)
                    losses_by_key[key][idx] = out['mse_reduced'][i]

                if coord.should_stop():
                    log.warning('Validation stopping because coord said so')
        except tf.errors.OutOfRangeError:
            log.info('Finished validation')

        mean_losses = [np.mean(losses) for losses in losses_by_key.values()]
        mean_loss = np.mean(mean_losses)

        # TODO threads are joined successfully but weird warnings about queues
        coord.request_stop()
        coord.join(threads)
        sess.close()



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
            outputs, ctx = self.model.build_model(provider, train=True)
            loss_op = self.model.calculate_loss(outputs, provider.targets, ctx['length'])

        if not val_provider is None:
            val_provider.batch_sequences_with_states(1,
                    collection=attend.GraphKeys.VAL_INPUT_RUNNERS)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                # tf.get_variable_scope().reuse_variables()
                val_outputs, val_ctx = self.model.build_model(val_provider, train=False)
                val_losses = self.model.calculate_losses(val_outputs,
                        val_provider.targets, val_ctx['length'], 'val_loss')

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
        train_sess  = tf.Session(graph=g)
        sess = train_sess
        sess.run(init_op)

        saver = tf.train.Saver()
        # Special input runners run separately because the supervisor can't
        # serialize them
        # input_threads = tf.train.start_queue_runners(sess=sess, coord=coord,
        #         collection='input_runners')
        input_threads = []
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
                    loss, _, summary, keys = sess.run([loss_op, train_op, summary_op, ctx['key']])
                    global_step_value = tf.train.global_step(sess, global_step)
                    log.debug('TRAIN %s - %s', global_step_value, loss)

                    summary_writer.add_summary(summary, global_step_value)

                    if coord.should_stop():
                        break
                    # END STEP

                if coord.should_stop():
                    break

                ## END OF EPOCH
                # SAVING
                save_path = saver.save(sess, log_dir + '/model.ckpt', global_step_value)

                # Validation after every epoch
                if val_provider:
                    self.test(
                            graph=g, saver=saver,
                            save_path=save_path,
                            provider=val_provider,
                            loss_ops=val_losses,
                            context_ops=val_ctx
                            )

            coord.request_stop()

        except tf.errors.OutOfRangeError:
            log.info('Done training -- epoch limit reached')
            notify('Done training', 'Took {:.1f}s'.format(time() - t_start))
        except Exception as e:
            log.critical(e)
            raise e
            notify('Error occurred', 'Took {:.1f}s'.format(time() - t_start))
        finally:
            log.debug('Finally - ...')
            # Requests the coordinator to stop, joins threads
            # and closes the summary writer if enabled through supervisor
            coord.join(threads + input_threads)
            # sv.stop()
            # coord.stop() DOESNT EXIST

        sess.close()
