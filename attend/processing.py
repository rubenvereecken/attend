import tensorflow as tf
import numpy as np

def extract_scalar(eventfile, key):
    def _read():
        for event in tf.train.summary_iterator(eventfile):
            for v in event.summary.value:
                if v.tag != key: continue
                yield v.simple_value
    return np.stack(_read())
