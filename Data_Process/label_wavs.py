#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: label_wavs.py
@Time: 2019/5/11 21:00
@Overview: The file is to label wav files and save or load those files with TFRecord data format.
"""
import argparse
import collections
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import pathlib
from scipy.io import wavfile
from scipy import signal
import sys,time
import IPython.display as display

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
from Data_Process import GenrateSpectrum

MAX_NUM_WAVS_PER_SPK = 2 ** 27 - 1  # ~134M
VOXCELEB_DIR = '../Dataset/wav'
TFRECORD_FILE = '../Dataset/wav.tfrecord'

TRAIN_VOXCELEB_DIR = '/data/voxceleb1/vox1_dev_wav'
TEST_VOXCELEB_DIR = '/data/voxceleb1/vox1_test_wav'
TRAIN_TFRECORD_FILE = 'wav.tfrecord'
TEST_TFRECORD_FILE = 'train_wav.bin'


def ensure_dir_exists(dir_name):
  """Makes sure the folder exists on disk.

  Args:
    dir_name: Path string to the folder we want to create.
  """
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# Create wav file list in retrain.py in tensorflow hub
def create_wav_list(wav_dir, test_precentage, validation_percentage):
    """Build a list of training wav files from the file system. Split wavs in the sub dir and return a data structure describing the lists of wavs for each label and their path.

    :param wav_dir: The root path to a folder containing sub folders of wavs
    :param test_precentage: Percentage of the wavs to reserve for test
    :param validation_percentage: Percentage of the wavs to reserve for validation

    :return: An orderedDict containing an entry for each label subfolder, with images split into training, testing and validation set within each label.
    """

    if not tf.gfile.Exists(wav_dir):
        tf.logging.error('Wav dir '+ wav_dir + 'not found.')
        return None
    result = collections.OrderedDict()
    sub_dirs = sorted(x[0] for x in tf.gfile.Walk(wav_dir))
    # Skip the root dir
    is_root_root = True
    for sub_dir in sub_dirs:
        if is_root_root:
            is_root_root = False
            continue
        extensions = sorted(set(os.path.normcase(ext)
                                for ext in ['wav', 'mp3', 'pcw', 'a-law']))
        file_list = []
        dir_name = os.path.basename(
            #tf.gfile.Walk() returns sub-dir with traling '\'
            sub_dir[:-1] if sub_dir.endswith('/') else sub_dir
        )

        if dir_name == wav_dir:
            continue
        tf.logging.info("Looking for wav files in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(wav_dir, dir_name, '*.' + extension)
            file_list.extend(tf.gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning('No wav files found')
            continue

        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_wavs = []
        testing_wavs = []
        validation_wavs = []
        for file_name in file_list:
            base_name = file_name
            file_path_split = base_name.split('\\')
            base_name = os.path.join(file_path_split[4], file_path_split[5])
            # Ignore anything after '_nohash_' in the file name
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # This looks a bit magical, but we need to decide whether this file should
            # go into the training, testing, or validation sets, and we want to keep
            # existing files in the same set even if more files are subsequently
            # added.
            # To do that, we need a stable way of deciding based on just the file name
            # itself, so we do a hash of that and then use that to generate a
            # probability value that we use to assign it.
            hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_WAVS_PER_SPK + 1)) *
                               (100.0 / MAX_NUM_WAVS_PER_SPK))
            if percentage_hash < validation_percentage:
                validation_wavs.append(base_name)
            elif percentage_hash < (test_precentage + validation_percentage):
                testing_wavs.append(base_name)
            else:
                training_wavs.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_wavs,
            'testing': testing_wavs,
            'validation': validation_wavs,
        }
        return result

# Create wav files list for TFRecord in Tensorflow tutorials
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def feature_map(label, value):
    """
    Return a float list from a float/double
    :param value:
    :return:
    """
    feature = {
        'label': _int64_feature(label),
        'spect': _bytes_feature(value)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

# Create tfrecord file for all wav files, the data format will be {label:lebel, spectrugram: spec}
def write_wav_tfrecord(wav_dir, tfrecord_file):
    if os.path.exists(tfrecord_file):
        print("File already existed!")
    data_root = pathlib.Path(wav_dir)

    all_wav_path = list(data_root.glob('*/*/*.wav'))
    all_wav_path = [str(path) for path in all_wav_path]

    spk_label = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    print(len(spk_label))

    # Add index for all speakers label
    label_to_index = dict((name, index) for index, name in enumerate(spk_label))

    all_wav_labels = [label_to_index[pathlib.Path(path).parent.parent.name]
                        for path in all_wav_path]
    wav_spec_list = np.zeros([len(all_wav_labels), 513, 300])

    # wav_path = all_wav_path[0]
    with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
        for index, wav_path in enumerate(all_wav_path):

            spec = GenrateSpectrum.GenerateSpect(wav_path)
            wav_spec_list[index]=spec
            spect_str = spec.tostring()

            feature_meta = feature_map(all_wav_labels[index], spect_str)
            writer.write(feature_meta.SerializeToString())

            # Progress Bar
            sys.stdout.write("\rProcessing Wav Data: [%s%s] %d%%" % ('#'*int(50*index/len(all_wav_path)), '='*(50 - int(50*index/len(all_wav_path))), 100*index/len(all_wav_path)))
            sys.stdout.flush()

    # for f0,f1 in feature_dataset:
    #     print(f0)
    #     print(f1)
    # print(wav_spec_list.shape())

def read_from_tfrecord(tfrecord_path):
    raw_wav_dataset = tf.data.TFRecordDataset(tfrecord_path)

    # Create a dictionary describing the wav spectrugram features.
    wav_feature_description = {
        'label': tf.FixedLenFeature([], tf.int64),
        'spect': tf.FixedLenFeature([], tf.string),
    }

    def _parse_spect_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.parse_single_example(example_proto, wav_feature_description)

    parsed_spect_dataset = raw_wav_dataset.map(_parse_spect_function)

    #return parsed_spect_dataset

    # for spect in parsed_spect_dataset:
    #     spectrugram = spect['spect'].numpy()
    #     label = spect['label']
    #     display.display(display.Image(data=spectrugram))

wav_lists = write_wav_tfrecord(VOXCELEB_DIR, TFRECORD_FILE)
read_from_tfrecord(TFRECORD_FILE)
