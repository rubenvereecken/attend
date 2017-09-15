import tensorflow as tf
import numpy as np
from time import time
from collections import OrderedDict


from .util import *
from attend.log import Log; log = Log.get_logger(__name__)
from attend import util, tf_util
import attend

class AttendSolver():
    def __init__(self, model, update_rule, learning_rate, stats_every):
        self.model = model

        self.update_rule = update_rule
        self.learning_rate = learning_rate
        self.summary_producer = None

        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        else:
            raise Exception()

        self.loss_names = ['mse', 'pearson_r', 'icc']
        from attend import SummaryProducer
        self.summary_producer = SummaryProducer(self.loss_names)
        self.stats_every = stats_every

        # TODO
        # This will contain the whole validation set, for losses
        # that are not implemented streaming
        self.placeholders_for_loss = {}


    def test(self, graph, saver, save_path, provider, init_op, context_ops, loss_ops,
             output_op,
            summary_writer, global_step):
        # Reasons for building a new session every validation step:
        # - There is no way to keep track of epochs OR to restart queues
        #   so a new session keeps it easy to loop through the input
        sess = tf.Session(graph=graph)
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,
                coord=coord, collection=attend.GraphKeys.VAL_INPUT_RUNNERS)

        # Load trained variables into the shared graph inside the val session
        saver.restore(sess, save_path)

        # These are batch losses per key
        loss_names = self.loss_names # TODO
        losses_by_loss_by_key = {k:OrderedDict() for k in loss_names}
        seq_lengths_by_key = OrderedDict()

        start = time.time()

        # TODO rid of this, memory expensive
        # Save the sequences so we can do an in-memory icc test
        predictions_by_key = OrderedDict()
        targets_by_key = OrderedDict()

        try:
            for i in range(1000000000):
                context_ops = context_ops.copy()
                context_ops.update(loss_ops['context'])
                # Note I used to also run loss_ops['all'] and loss_ops['batch'],
                # I think for sequence-wise losses
                # They've all currently been replaced with streaming 'total'
                # losses
                ctx, total_loss, predictions, targets = sess.run(
                    [context_ops, loss_ops['total'],
                     output_op, provider.targets])
                keys = list(map(lambda x: x.decode(), ctx['key']))

                for i, key in enumerate(keys):
                    # if key not in seq_lengths_by_key:
                    #     seq_lengths_by_key[key] = ctx['length'][i]

                    predictions_bits = predictions_by_key.get(key, [])
                    targets_bits = targets_by_key.get(key, [])
                    predictions_bits.append(predictions[i])
                    targets_bits.append(targets[i])
                    predictions_by_key[key] = predictions_bits
                    targets_by_key[key] = targets_bits

                if coord.should_stop():
                    log.warning('Validation stopping because coord said so')
        except tf.errors.OutOfRangeError:
            log.info('Finished validation in %.2fs', time.time() - start)

        seq_lengths_by_key = OrderedDict(zip(ctx['all_keys'],
                                             ctx['all_lengths']))

        mean_by_loss = {}
        # mean_by_loss = { k: np.mean(v) for k, v in all_loss.items() }
        mean_by_loss.update(total_loss)
        # n_keys = len(ctx['all_keys'])
        # for k, v in total_loss.items():
        #     all_loss.update({ k: np.zeros((n_keys, *v.shape)) })

        # Compute icc manually
        def _piece_together(d):
            return OrderedDict((k, np.concatenate(v)) for k, v in d.items())
        max_length = max(seq_lengths_by_key.values())
        predictions_by_key = _piece_together(predictions_by_key)
        targets_by_key = _piece_together(targets_by_key)
        max_padded_length = max(map(lambda v: v.shape[0], targets_by_key.values()))

        from attend.util import pad_and_stack
        predictions = pad_and_stack(predictions_by_key.values())
        targets = pad_and_stack(targets_by_key.values())
        del predictions_by_key, targets_by_key

        icc_op = loss_ops['icc']
        icc_score = sess.run(icc_op, {
            self.placeholders_for_loss['predictions']: predictions,
            self.placeholders_for_loss['targets']: targets,
            self.placeholders_for_loss['lengths']: list(seq_lengths_by_key.values()),
        })
        mean_by_loss['icc'] = icc_score
        del predictions, targets

        summary = self.summary_producer.create_loss_summary(sess, mean_by_loss)
                # all_loss)
        summary_writer.add_summary(summary, global_step)

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
              save_eval_graph=True,
              restore_if_possible=True,
              keep_all_checkpoints=False,
              show_progress_bar=None):

        if show_progress_bar is None and debug is False:
            show_progress_bar = True
        if show_progress_bar:
            from tqdm import tqdm
            progress_wrapper = tqdm
        else:
            progress_wrapper = lambda i, **kwargs: i

        total_steps = num_epochs * steps_per_epoch
        g = tf.get_default_graph()

        # For now, its results are stored inside the provider
        provider.batch_sequences_with_states(is_training=True, reuse=False)

        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Prediction and loss
        with tf.variable_scope(tf.get_variable_scope()):
            out, ctx = self.model.build_model(provider, True, total_steps)
            outputs = out['output']
            loss_op = self.model.calculate_loss(outputs, provider.targets, ctx['length'])

        if not val_provider is None:
            n_vars = len(tf.trainable_variables())

            val_provider.batch_sequences_with_states(1, is_training=False,
                    reuse=True,
                    collection=attend.GraphKeys.VAL_INPUT_RUNNERS)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                # tf.get_variable_scope().reuse_variables()
                val_out, val_ctx = self.model.build_model(val_provider, False)
                val_outputs = val_out['output']

            # Should not reuse variables, so back out of the reusing scope
            val_losses, _ = self.model.calculate_losses(val_outputs,
                    val_provider.targets, val_ctx['key'], val_ctx['length'], 'val_loss')
            loss_predictions = tf.placeholder(tf.float32, [None, None, 1])
            loss_targets = tf.placeholder(tf.float32, [None, None, 1])
            loss_lengths = tf.placeholder(tf.int32, [None])
            self.placeholders_for_loss.update(dict(predictions=loss_predictions,
                                               targets=loss_targets,
                                               lengths=loss_lengths))
            from .losses import icc
            icc_loss = self.model.calculate_loss(loss_predictions, loss_targets,
                                                 loss_lengths, icc(3,1))
            val_losses['icc'] = icc_loss

            if debug:
                assert n_vars == len(tf.trainable_variables()), 'New vars were created for val'

        if save_eval_graph:
            log.info('Creating eval graph')
            eval_graph = self.create_test_graph(**provider.__dict__)
            # tf.train.write_graph(eval_graph, log_dir, 'eval_model.graph.proto',
            #         as_text=False)
            # These saveables prevent the graph from being reconstructed so remove
            # for serialization
            saveables = eval_graph.get_collection_ref('saveable_objects')
            backup = saveables.copy()
            saveables.clear()
            tf.train.export_meta_graph(log_dir + '/eval_model.meta.proto',
                    graph=eval_graph, as_text=False, clear_devices=True)
            eval_graph.get_collection_ref('saveable_objects').extend(backup)
            log.info('Exported eval_model')


        with tf.variable_scope('optimizer', reuse=False):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            # train_op = optimizer.minimize(loss_op, global_step=global_step,
            #         var_list=tf.trainable_variables())
            grads = tf.gradients(loss_op, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))

            # All updates that aren't part of the graph
            # Currently just batch norm moving averages
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars,
                        global_step=global_step)
            num_vars = np.sum(list(map(lambda v: np.prod(v.shape), tf.trainable_variables())))
            log.info('Total trainable vars {}'.format(num_vars))

        # Initialize variables
        # The Supervisor actually runs this automatically, but I'm keeping this
        init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

        # Summary op
        tf.summary.scalar('batch_loss', loss_op, family='train')
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name, var)
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
        # from tensorflow.python import debug as tf_debug
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess, thread_name_filter="MainThread$")
        sess.run(init_op)

        saver = tf.train.Saver(save_relative_paths=True,
                               max_to_keep=num_epochs if keep_all_checkpoints else 2)

        if restore_if_possible:
            states = tf.train.get_checkpoint_state(log_dir)
            if not states is None:
                checkpoint_paths = states.all_model_checkpoint_paths
                last_checkpoint = tf.train.latest_checkpoint(log_dir)
                log.info('Resuming training from checkpoint {}'.format(last_checkpoint))
                saver.restore(sess, last_checkpoint)
                saver.recover_last_checkpoints(checkpoint_paths)

        # Special input runners run separately because the supervisor can't
        # serialize them
        input_threads = tf.train.start_queue_runners(sess=sess, coord=coord,
                                                     collection='input_runners')
        # input_threads = []
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        g.finalize() # No ops can be added after this

        log.info('Started training')
        losses = np.empty(self.stats_every) # Keep losses to average every so often
        global_step_value = sess.run(global_step)
        t_start = time.time()
        t_stats = time.time()

        try:
            # while not coord.should_stop():
            while global_step_value < num_epochs * steps_per_epoch:
                step_i = global_step_value % num_epochs
            # for epoch_i in progress_wrapper(range(num_epochs)):
            #     for step_i in progress_wrapper(range(steps_per_epoch)):
                if (global_step_value) % self.stats_every == 0:
                    t_stats = time.time()
                try:
                    loss, _, summary, keys = sess.run([loss_op, train_op,
                                                        summary_op, ctx['key']])
                    # np.savez('{}/output.{:05d}'.format(log_dir, global_step_value),
                    #          output=outputs_arr,target=targets_arr)
                    losses[step_i % self.stats_every] = loss # Circular buffer
                    # keys = list(map(lambda x: x.decode(), keys))
                # If duplicate key is encountered this could happen rarely
                except tf.errors.InvalidArgumentError as e:
                    # log.exception(e)
                    raise e
                global_step_value = tf.train.global_step(sess, global_step)
                log.debug('TRAIN %s - %s', global_step_value, loss)

                summary_writer.add_summary(summary, global_step_value)

                # Runtime stats every so often
                if global_step_value % self.stats_every == 0:
                    stats_summary = self.summary_producer.create_stats_summary(
                            sess, time.time() - t_stats, global_step_value,
                            np.mean(losses))
                    summary_writer.add_summary(stats_summary, global_step_value)


                if coord.should_stop():
                    break
                # END STEP

                if global_step_value % steps_per_epoch == 0:
                    ## END OF EPOCH
                    # SAVING
                    save_path = saver.save(sess, log_dir + '/model.ckpt',
                                           global_step_value, write_meta_graph=False)

                    # Validation after every epoch
                    if val_provider:
                        self.test(
                                graph=g, saver=saver,
                                save_path=save_path,
                                provider=val_provider,
                                init_op=init_op,
                                loss_ops=val_losses,
                                context_ops=val_ctx,
                                output_op=val_outputs,
                                summary_writer=summary_writer,
                                global_step = global_step_value
                                )

            coord.request_stop()

        except tf.errors.OutOfRangeError:
            log.info('Done training -- epoch limit reached')
            notify('Done training', 'Took {:.1f}s'.format(time.time() - t_start))
        except Exception as e:
            log.exception(e)
            notify('Error occurred', 'Took {:.1f}s'.format(time.time() - t_start))
        finally:
            log.debug('Joining threads - ...')
            # Requests the coordinator to stop, joins threads
            # and closes the summary writer if enabled through supervisor
            coord.join(threads + input_threads)
            # sv.stop()
            # coord.stop() DOESNT EXIST

        sess.close()


    def create_test_graph(self, **kwargs):
        from attend.provider import InMemoryProvider, Provider

        graph = tf.Graph()
        provider = InMemoryProvider(kwargs.pop('feature_dims'),
                                    **util.pick(kwargs, util.params_for(Provider.__init__)))
        scope = ''

        with graph.as_default():
            with tf.variable_scope(scope, reuse=False):
                provider.batch_sequences_with_states(is_training=False, reuse=False)
                out_ops, ctx_ops = self.model.build_model(provider, False)
                reset_op = provider.state_saver.reset_states()
                tf_util.add_to_collection(attend.GraphKeys.STATE_RESET, reset_op, graph)

        return graph
