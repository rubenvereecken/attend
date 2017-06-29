import tensorflow as tf

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
              log_dir,
              debug=False):

        # For now, its results are stored inside the provider
        provider.batch_sequences_with_states()
        # images, targets = input_pipeline([filename], batch_size,
        #         time_steps=time_steps, num_epochs=num_epochs)

        # Build a graph that computes the loss
        # loss = self.model.build_model(images, targets)


        # with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        loss_op = self.model.build_model(provider)


        with tf.name_scope('optimizer'):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            grads = tf.gradients(loss_op, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        # Initialize variables
        init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())


        # from tensorflow.python import debug as tf_debug
        # Create a session for running operations in the Graph.
        # sess = tf.Session()

        # Initialize the variables (like the epoch counter).

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        sv = tf.train.Supervisor(logdir=log_dir)
        coord = sv.coord
        config = tf.ConfigProto(
            intra_op_parallelism_threads=4
        )
        with sv.managed_session(config=config) as sess:
        # with tf.Session() as sess:
            # TODO write down that init is no longer needed
            # sess.run(init_op)
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                while not sv.should_stop():
                    loss, _ = sess.run([loss_op, train_op])
                    print('TEST')
                    print(loss)
                    # Run training steps or whatever
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                print('AHM DONE')
                # When done, ask the threads to stop.
                sv.request_stop()

            # Wait for threads to finish.
            coord.join(threads)
            sess.close()
            print('Finished', loss)
            import sys
            sys.stdout.flush()
