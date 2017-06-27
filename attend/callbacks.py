from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
from collections import Iterable
from collections import OrderedDict
import csv
import json
import os
import time
import warnings

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras import callbacks
from tensorflow.python.ops import array_ops
from tensorflow.python.summary import summary as tf_summary
from tensorflow.python.training import saver as saver_lib


class TensorBoard(callbacks.Callback):
  # pylint: disable=line-too-long
  """Tensorboard basic visualizations.

  This callback writes a log for TensorBoard, which allows
  you to visualize dynamic graphs of your training and test
  metrics, as well as activation histograms for the different
  layers in your model.

  Arguments:
      log_dir: the path of the directory where to save the log
          files to be parsed by Tensorboard.
      histogram_freq: frequency (in epochs) at which to compute activation
          histograms for the layers of the model. If set to 0,
          histograms won't be computed.
      write_graph: whether to visualize the graph in Tensorboard.
          The log file can become quite large when
          write_graph is set to True.
      write_images: whether to write model weights to visualize as
          image in Tensorboard.
      embeddings_freq: frequency (in epochs) at which selected embedding
          layers will be saved.
      embeddings_layer_names: a list of names of layers to keep eye on. If
          None or empty list all the embedding layer will be watched.
      embeddings_metadata: a dictionary which maps layer name to a file name
          in which metadata for this embedding layer is saved. See the
          [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
          about metadata files format. In case if the same metadata file is
          used for all embedding layers, string can be passed.
  """

  # pylint: enable=line-too-long

  def __init__(self,
               log_dir='./logs',
               histogram_freq=0,
               write_graph=True,
               write_images=False,
               embeddings_freq=0,
               embeddings_layer_names=None,
               embeddings_metadata=None):
    super(TensorBoard, self).__init__()
    self.log_dir = log_dir
    self.histogram_freq = histogram_freq
    self.merged = None
    self.write_graph = write_graph
    self.write_images = write_images
    self.embeddings_freq = embeddings_freq
    self.embeddings_layer_names = embeddings_layer_names
    self.embeddings_metadata = embeddings_metadata or {}

  def set_model(self, model):
    self.model = model
    self.sess = K.get_session()
    if self.histogram_freq and self.merged is None:
      for layer in self.model.layers:

        for weight in layer.weights:
          tf_summary.histogram(weight.name, weight)
          if self.write_images:
            w_img = array_ops.squeeze(weight)
            shape = w_img.get_shape()
            if len(shape) > 1 and shape[0] > shape[1]:
              w_img = array_ops.transpose(w_img)
            if len(shape) == 1:
              w_img = array_ops.expand_dims(w_img, 0)
            w_img = array_ops.expand_dims(array_ops.expand_dims(w_img, 0), -1)
            tf_summary.image(weight.name, w_img)

        if hasattr(layer, 'output'):
          tf_summary.histogram('{}_out'.format(layer.name), layer.output)
    self.merged = tf_summary.merge_all()

    if self.write_graph:
      self.writer = tf_summary.FileWriter(self.log_dir, self.sess.graph)
    else:
      self.writer = tf_summary.FileWriter(self.log_dir)

    if self.embeddings_freq:
      self.saver = saver_lib.Saver()

      embeddings_layer_names = self.embeddings_layer_names

      if not embeddings_layer_names:
        embeddings_layer_names = [
            layer.name for layer in self.model.layers
            if type(layer).__name__ == 'Embedding'
        ]

      embeddings = {
          layer.name: layer.weights[0]
          for layer in self.model.layers if layer.name in embeddings_layer_names
      }

      embeddings_metadata = {}

      if not isinstance(self.embeddings_metadata, str):
        embeddings_metadata = self.embeddings_metadata
      else:
        embeddings_metadata = {
            layer_name: self.embeddings_metadata
            for layer_name in embeddings.keys()
        }

      config = projector.ProjectorConfig()
      self.embeddings_logs = []

      for layer_name, tensor in embeddings.items():
        embedding = config.embeddings.add()
        embedding.tensor_name = tensor.name

        self.embeddings_logs.append(
            os.path.join(self.log_dir, layer_name + '.ckpt'))

        if layer_name in embeddings_metadata:
          embedding.metadata_path = embeddings_metadata[layer_name]

      projector.visualize_embeddings(self.writer, config)

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}

    if self.validation_data and self.histogram_freq:
      if epoch % self.histogram_freq == 0:
        # TODO(fchollet): implement batched calls to sess.run
        # (current call will likely go OOM on GPU)
        if self.model.uses_learning_phase:
          cut_v_data = len(self.model.inputs)
          val_data = self.validation_data[:cut_v_data] + [0]
          tensors = self.model.inputs + [K.learning_phase()]
        else:
          val_data = self.validation_data
          tensors = self.model.inputs
        feed_dict = dict(zip(tensors, val_data))
        result = self.sess.run([self.merged], feed_dict=feed_dict)
        summary_str = result[0]
        self.writer.add_summary(summary_str, epoch)

    if self.embeddings_freq and self.embeddings_logs:
      if epoch % self.embeddings_freq == 0:
        for log in self.embeddings_logs:
          self.saver.save(self.sess, log, epoch)

    for name, value in logs.items():
      if name in ['batch', 'size']:
        continue
      summary = tf_summary.Summary()
      summary_value = summary.value.add()
      summary_value.simple_value = value.item()
      summary_value.tag = name
      self.writer.add_summary(summary, epoch)
    self.writer.flush()

  def on_train_end(self, _):
    self.writer.close()
