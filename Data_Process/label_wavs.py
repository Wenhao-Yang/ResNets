#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: label_wavs.py
@Time: 2019/5/11 21:00
@Overview:
"""
import argparse
import collections
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys

import numpy as np
import tensorflow as tf

MAX_NUM_WAVS_PER_SPK = 2 ** 27 - 1  # ~134M

def ensure_dir_exists(dir_name):
  """Makes sure the folder exists on disk.

  Args:
    dir_name: Path string to the folder we want to create.
  """
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

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

voxceleb_dir = '../Dataset/wav'
create_wav_list(voxceleb_dir, 20, 20)



def get_wav_path(wav_lists, label_name, index, wav_dir, category):
  """Returns a path to an radio for a label at the given index.

  :param wav_lists: OrderedDict of training wavs for each label.
  :param label_name: Label string we want to get an wav for.
  :param index: Int offset of the wav we want. This will be moduloed by the available number of wavs for the label, so it can be arbitrarily large.
  :param wav_dir: Root folder string of the sub folders containing the training wavs.
  :param category: Name string of set to pull wavs from - training, testing, or
    validation.
  :return: File system path string to an image that meets the requested parameters.
  """

  if label_name not in wav_lists:
    tf.logging.fatal('Label does not exist %s.', label_name)
  label_lists = wav_lists[label_name]
  if category not in label_lists:
    tf.logging.fatal('Category does not exist %s.', category)
  category_list = label_lists[category]
  if not category_list:
    tf.logging.fatal('Label %s has no wavs in the category %s.',
                     label_name, category)
  mod_index = index % len(category_list)
  base_name = category_list[mod_index]
  sub_dir = label_lists['dir']
  full_path = os.path.join(wav_dir, sub_dir, base_name)
  return full_path