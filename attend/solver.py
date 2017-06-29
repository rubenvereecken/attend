import tensorflow as tf
from attend.provider import input_pipeline
# from sewa.model import SEWAModel

class AttendSolver():
    def __init__(self, model, update_rule, learning_rate):
        self.model = model

        self.update_rule = update_rule
        self.learning_rate = learning_rate

        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        else:
            raise Exception()


    def train(self, data_file, num_epochs, batch_size, time_steps,
              log_dir,
              debug=False):
        filename = data_file
        images, targets = input_pipeline([filename], batch_size,
                time_steps=time_steps, num_epochs=num_epochs)

        # Build a graph that computes the loss
        # loss = self.model.build_model(images, targets)


        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            loss_op = self.model.build_model(images, targets)
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
                while not coord.should_stop():
                    loss, _ = sess.run([loss_op, train_op])
                    print(loss)
                    break
                    # Run training steps or whatever
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()
            # if debug:
            #     sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            # Wait for threads to finish.
            coord.join(threads)
            sess.close()
