#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: voxceleb_speaker_main.py
@Time: 19-5-22 下午3:50
@Overview: Using the data collected from the Data process to train a ResNet model for speaker recognition.
"""
import tensorflow as tf
import os
from Data_Process.label_wavs import TRAIN_TFRECORD_FILE, TEST_TFRECORD_FILE
from ResNets import resnet_run_loop
from ResNets import resnet_model
from absl import app as absl_app
from absl import flags
from ResNets.utils.flags import core as flags_core
from ResNets.utils.logs import logger

HEIGHT = 513        # The height of spectrugram array
WIDTH = 300         # The width of spectrugram array
NUM_CHANNELS = 1
NUM_CLASSES = 1211  # TODO 1251 is the total number of spks in Voxceleb1
DATASET_NAME = 'VOXCELEB1'
_SHUFFLE_BUFFER = 6400
########################################
# Data Processing
########################################
# TODO:
NUM_WAVS = {
    'train': 118914,
    'validation': 29728,
}

def get_filenames(is_training, data_dir):
    """Returns a list of filenames"""
    assert tf.gfile.Exists(data_dir), ("Run the label_wavs.py to pre-process wav file")

    if is_training:
        return [
            os.path.join(data_dir, TRAIN_TFRECORD_FILE)
        ]
    else:
        return  [os.path.join(data_dir, TEST_TFRECORD_FILE)]

def parse_record(raw_record, is_training, dtype):
    """Parse Voxceleb wav spectrugrams and label from a raw record"""
    # wav_feature_description = {
    #     'label': tf.FixedLenFeature([], tf.int64),
    #     'spect': tf.FixedLenFeature([], tf.string),
    # }
    #
    # def _parse_spect_function(example_proto):
    #     # Parse the input tf.Example proto using the dictionary above.
    #     return tf.parse_single_example(example_proto, wav_feature_description)
    #
    # record_vector = _parse_spect_function(raw_record)
    label = raw_record['label']

    spect = tf.io.decode_raw(raw_record['spect'], tf.uint8)

    # The first byte represents the label, which we convert from uint8 to int32
    # and then to one-hot.
    label = tf.cast(label, tf.int32)


    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(spect,
                             [NUM_CHANNELS, HEIGHT, WIDTH])

    # Convert from [depth, height, width] to [height, width, depth], and cast as
    # float32.
    image = tf.cast(tf.transpose(a=depth_major, perm=[1, 2, 0]), tf.float32)

    image = tf.cast(image, dtype)

    return image, label

    print(str(raw_record['label']))
    #label = tf.cast(raw_record['label'], tf.int32)
    #spect = tf.cast(raw_record['spect'].numpy(), dtype)

    # return spect, spect

def preprocess_spect(spect, is_training):
    """Preprocess a spectrugram of [height, width]"""
    if is_training:
        # TODO resize the spectrugram
        spect = spect

    # TODO Mean and variance normalisation
    spect = spect
    return spect

def input_fn(is_training,
             data_dir,
             batch_size,
             num_epochs=1,
             dtype=tf.float32,
             datasets_num_private_threads=None,
             num_parallel_batches=1,
             parse_record_fn=parse_record,
             input_context=None):
    """Input function which provides batches for train or eval.
    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: The directory containing the input data.
      batch_size: The number of samples per batch.
      num_epochs: The number of epochs to repeat the dataset.
      dtype: Data type to use for features
      datasets_num_private_threads: Number of private threads for tf.data.
      num_parallel_batches: Number of parallel batches for tf.data.
      parse_record_fn: Function to use for parsing the records.
      input_context: A `tf.distribute.InputContext` object passed in by
        `tf.distribute.Strategy`.
    Returns:
      A dataset that can be used for iteration.
    """
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.TFRecordDataset(filenames)

    if input_context:
        tf.compat.v1.logging.info(
            'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d' % (
                input_context.input_pipeline_id, input_context.num_input_pipelines))
        dataset = dataset.shard(input_context.num_input_pipelines,
                                input_context.input_pipeline_id)
    #if is_training:
    # Shuffle the input files
    #    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

    return resnet_run_loop.process_record_dataset(
        dataset=dataset,
        is_training=is_training,
        batch_size=batch_size,
        shuffle_buffer=_SHUFFLE_BUFFER,
        parse_record_fn=parse_record,
        num_epochs=num_epochs,
        dtype=dtype,
        datasets_num_private_threads=datasets_num_private_threads,
        num_parallel_batches=num_parallel_batches
    )

def get_synth_input_fn(dtype):
  return resnet_run_loop.get_synth_input_fn(
      HEIGHT, WIDTH, NUM_CHANNELS, NUM_CLASSES, dtype=dtype)


###############################################################################
# Running the model
###############################################################################
class ResNetModel(resnet_model.Model):
  """Model class with appropriate for Voxceleb1 data."""

  def __init__(self, resnet_size, data_format=None, num_classes=NUM_CLASSES,
               resnet_version=resnet_model.DEFAULT_VERSION,
               dtype=resnet_model.DEFAULT_DTYPE):
    """These are the parameters that work for dataset.
    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
      to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.
    Raises:
      ValueError: if invalid resnet_size is chosen
    """

    # TODO remove the checking process, which seems to be unecessary.
    # if resnet_size % 6 != 2:
    #   raise ValueError('resnet_size must be 6n + 2:', resnet_size)

    # TODO the block should be 8 for ResNet-18
    num_blocks = (resnet_size - 2) // 6
    # num_blocks = 4

    # TODO define of ResNet is not correct.
    super(ResNetModel, self).__init__(
        resnet_size=resnet_size,
        bottleneck=False,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=3,
        conv_stride=1,
        first_pool_size=None,
        first_pool_stride=None,
        block_sizes=[num_blocks] * 3,
        block_strides=[1, 2, 2],
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype
    )


def ResNet_model_fn(features, labels, mode, params):
  """Model function for ResNet."""
  features = tf.reshape(features, [-1, HEIGHT, WIDTH, NUM_CHANNELS])

  # Learning rate schedule follows arXiv:1512.03385 for ResNet-56 and under.
  learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
      batch_size=params['batch_size'] * params.get('num_workers', 1),
      batch_denom=128, num_images=NUM_WAVS['train'],
      boundary_epochs=[91, 136, 182], decay_rates=[1, 0.1, 0.01, 0.001])

  # Weight decay of 2e-4 diverges from 1e-4 decay used in the ResNet paper
  # and seems more stable in testing. The difference was nominal for ResNet-56.
  weight_decay = 2e-4

  # Empirical testing showed that including batch_normalization variables
  # in the calculation of regularized loss helped validation accuracy
  # for the CIFAR-10 dataset, perhaps because the regularization prevents
  # overfitting on the small data set. We therefore include all vars when
  # regularizing and computing loss during training.
  def loss_filter_fn(_):
    return True

  return resnet_run_loop.resnet_model_fn(
      features=features,
      labels=labels,
      mode=mode,
      model_class=ResNetModel,
      resnet_size=params['resnet_size'],
      weight_decay=weight_decay,
      learning_rate_fn=learning_rate_fn,
      momentum=0.9,
      data_format=params['data_format'],
      resnet_version=params['resnet_version'],
      loss_scale=params['loss_scale'],
      loss_filter_fn=loss_filter_fn,
      dtype=params['dtype'],
      fine_tune=params['fine_tune']
  )


def define_resnet_flags():
    # Define the flags during the running
    resnet_run_loop.define_resnet_flags()
    flags.adopt_module_key_flags(resnet_run_loop)
    flags_core.set_defaults(data_dir='../Dataset',
                          model_dir='model',
                          resnet_size='56',
                          train_epochs=30,
                          epochs_between_evals=10,
                          batch_size=64,
                          spect_bytes_as_serving_input=False)


def run_resnet(flags_obj):
    """Run ResNet training and eval loop.
    Args:
    flags_obj: An object containing parsed flag values.
    Returns:
    Dictionary of results. Including final accuracy.
    """
    # The flag that indicates the input is bytes
    if flags_obj.spect_bytes_as_serving_input:
        tf.compat.v1.logging.fatal(
            '--spect_bytes_as_serving_input cannot be set to True for VOXCELEB. '
            'This flag is only applicable to ImageNet.')
        return

    # Define input function for data input
    # If the use_synthetic_data flag is True, then use the synth_input_fn to provide input data.
    # Otherwise, input_fn gives input data.
    input_function = (flags_obj.use_synthetic_data and
                    get_synth_input_fn(flags_core.get_tf_dtype(flags_obj)) or
                    input_fn)

    result = resnet_run_loop.resnet_main(
      flags_obj, ResNet_model_fn, input_function, DATASET_NAME,
      shape=[HEIGHT, WIDTH, NUM_CHANNELS])

    return result


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_resnet(flags.FLAGS)


if __name__ == '__main__':
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  define_resnet_flags()
  absl_app.run(main)
