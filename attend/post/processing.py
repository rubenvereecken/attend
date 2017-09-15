import tensorflow as tf
import numpy as np

def extract_scalar(event_files, key):
    def _read(eventfile):
        try:
            for event in tf.train.summary_iterator(eventfile):
                for v in event.summary.value:
                    if v.tag != key: continue
                    yield v.simple_value
        except Exception as e:
            pass


    def _read_all():
        for eventfile in event_files:
            event_values = np.array(list(_read(eventfile)))
            if event_values.shape == ():
                print('No values found in', eventfile)
                continue
            yield event_values

    return np.concatenate(list(_read_all()))


def get_tags(eventfile, max_it=1020):
    event_it = tf.train.summary_iterator(eventfile)
    tags = set()

    for i in range(max_it):
        event = next(event_it)
        for v in event.summary.value:
            tags.add(v.tag)

    return list(tags)


def find_tfevents_file(log_dir):
    import glob
    # Sorted by timestamp
    event_files = sorted(glob.glob(log_dir + '/*tfevents*'))
    return event_files
