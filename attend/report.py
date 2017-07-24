import tensorflow as tf
import numpy as np

import attend

class SummaryProducer:

    def __init__(self, loss_names):
        self.loss_names = loss_names

        self.mean_losses = { k: tf.placeholder(tf.float32, ()) for k in self.loss_names }
        self.mean_hists = { k: tf.placeholder(tf.float32, (None)) for k in self.loss_names }
        self.mean_hists_norm = { k: tf.placeholder(tf.float32, (None)) for k in self.loss_names }

        self.summary_op = self.build_summary_op()

    def mean_scalar(self):
        mean_summaries = [ tf.summary.scalar(loss_name + '/mean',
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
        mean_scalars = self.mean_scalar()
        mean_hists = self.mean_hist()
        all_summaries = mean_scalars + mean_hists

        summary_op = tf.summary.merge(all_summaries)

        return summary_op

    def write_summary(self, sess, mean_loss, mean_by_seq, mean_by_seq_norm):
        feed_dict = {}

        for k, v in mean_loss.items():
            feed_dict[self.mean_losses[k]] = v

        for k, v in mean_by_seq.items():
            feed_dict[self.mean_hists[k]] = v

        for k, v in mean_by_seq_norm.items():
            feed_dict[self.mean_hists_norm[k]] = v

        summary = sess.run(self.summary_op, feed_dict=feed_dict)
        return summary
