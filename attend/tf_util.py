import tensorflow as tf

def add_to_collection(key, values, graph=None):
    if not isinstance(values, list):
        values = [values]

    if graph is None:
        graph = tf.get_default_graph()

    for v in values:
        graph.add_to_collection(key, v)

    return graph


def name(t):
    return ':'.join(t.name.split('/')[-1].split(':')[:-1])


def get_collection_as_dict(key, graph=None):
    if graph is None:
        graph = tf.get_default_graph()
    collection = graph.get_collection(key)
    collection = { name(v): v for v in collection }
    return collection