import tensorflow as tf
import numpy as np

def streaming_mse(predictions, targets, keys, lengths, mask=None):
    with tf.name_scope('streaming_mse'):
        totals = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string,
                    value_dtype=tf.float32, default_value=0,
                    name='total_table')
        counts = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string,
                    value_dtype=tf.float32, default_value=0,
                    name='count_table')
        mse = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string,
                    value_dtype=tf.float32, default_value=0,
                    name='mse_table')

        total_se = totals.lookup(keys)
        masked_diff = (predictions - targets) * mask
        current_se = tf.reduce_sum(tf.square(masked_diff, name='mse'), axis=1)
        total_se += current_se
        total_update = totals.insert(keys, total_se)
        count = counts.lookup(keys)
        count += tf.cast(lengths, tf.float32)
        count_update = counts.insert(keys, count)
        mse_update = mse.insert(keys, total_se / (count))

    with tf.control_dependencies([total_update, count_update, mse_update]):
        return mse.lookup(keys), mse
