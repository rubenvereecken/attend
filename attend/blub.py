import attend
from attend import Evaluator

import numpy as np

e = Evaluator.init_from_logs('/vol/bitbucket/rv1017/log/sigmoid_31-07-2017-14-53-13')

reader = attend.readers.HDF5SequenceReader('/home/ruben/tmp/confer-splits/val.hdf5', 'conflict')
key = reader.keys[0]
features = reader.features[key]
features = np.reshape(features, (-1, 272))
target = reader.targets[key]
n = len(target)

out = e.evaluate(features)
outputs = out['output']
