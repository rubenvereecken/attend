import tensorflow as tf
import numpy as np

from time import time

import attend

class SummaryProducer:
    def __init__(self, loss_names):
        self.loss_names = loss_names

        self.mean_losses = { k: tf.placeholder(tf.float32, ()) for k in self.loss_names }
        self.mean_hists = { k: tf.placeholder(tf.float32, (None)) for k in self.loss_names }
        self.mean_hists_norm = { k: tf.placeholder(tf.float32, (None)) for k in self.loss_names }
        self.summary_op = self.build_summary_op()

        self.last_step = 0
        self.steps_per_second = tf.placeholder(tf.float32, ())
        self.mem_usage = tf.placeholder(tf.float32, ())
        self.mean_train_loss = tf.placeholder(tf.float32, ())
        self.stats_op = self.build_stats_op()

        import os
        import psutil
        self.process = psutil.Process(os.getpid())

    def mean_scalar(self):
        mean_summaries = [ tf.summary.scalar(loss_name + '/total',
            self.mean_losses[loss_name],
            collections=attend.GraphKeys.VAL_SUMMARIES,
            family='validation') for loss_name in self.loss_names]

        return mean_summaries

    def mean_hist(self):
        mean_hists = [ tf.summary.histogram(loss_name + '/mean_hist',
            self.mean_hists[loss_name],
            collections=attend.GraphKeys.VAL_SUMMARIES,
            family='validation') for loss_name in self.loss_names ]

        mean_hists_norm = [ tf.summary.histogram(loss_name + '/mean_norm',
            self.mean_hists_norm[loss_name],
            collections=attend.GraphKeys.VAL_SUMMARIES,
            family='validation') for loss_name in self.loss_names ]

        return mean_hists + mean_hists_norm

    def build_summary_op(self):
        start = time()
        mean_scalars = self.mean_scalar()
        mean_hists = self.mean_hist()
        all_summaries = mean_scalars + mean_hists

        summary_op = tf.summary.merge(all_summaries)
        # self.summary_op = summary_op
        from attend.log import Log; log = Log.get_logger(__name__)
        log.debug('Built summary test summary op in %.3f s', time() - start)

        return summary_op

    def create_loss_summary(self, sess, mean_loss, mean_by_seq, mean_by_seq_norm=None):
        feed_dict = {}
        # TODO fix this some day
        if mean_by_seq_norm is None:
            mean_by_seq_norm = {}
            for k, v in mean_by_seq.items():
                mean_by_seq_norm[k] = np.zeros_like(v)

        for k, v in mean_loss.items():
            feed_dict[self.mean_losses[k]] = v

        for k, v in mean_by_seq.items():
            feed_dict[self.mean_hists[k]] = v

        for k, v in mean_by_seq_norm.items():
            feed_dict[self.mean_hists_norm[k]] = v

        summary = sess.run(self.summary_op, feed_dict=feed_dict)
        return summary

    def build_stats_op(self):
        scalars = [
                tf.summary.scalar('steps_per_second',
                    self.steps_per_second,
                    collections=attend.GraphKeys.STATS_SUMMARIES,
                    family='train'),
                tf.summary.scalar('mem_usage',
                    self.mem_usage,
                    collections=attend.GraphKeys.STATS_SUMMARIES,
                    family='train'),
                tf.summary.scalar('mean_loss',
                    self.mean_train_loss,
                    collections=attend.GraphKeys.STATS_SUMMARIES,
                    family='train')
                ]
        summary_op = tf.summary.merge(scalars)
        return summary_op

    def create_stats_summary(self, sess, elapsed, global_step, mean_loss):
        steps = global_step - self.last_step
        steps_per_second = steps / elapsed

        mem_usage = self.process.memory_info().rss
        mem_usage *= 1e-6

        feed_dict = {
                self.steps_per_second: steps_per_second,
                self.mem_usage: mem_usage,
                self.mean_train_loss: mean_loss,
                }
        summary = sess.run(self.stats_op, feed_dict=feed_dict)
        self.last_step = global_step
        return summary
