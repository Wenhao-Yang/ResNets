#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: Voxceleb.py
@Time: 19-5-23 下午4:57
@Overview: Implement Dataset for Voxceleb
"""
from torch.utils.data import Dataset
from Data_Process import GenrateSpectrum
import numpy as np
import os
import pathlib

def default_loader(wav):
    spec = GenrateSpectrum.GenerateSpect(wav)
    return spec


class voxceleb(Dataset):
    def __init__(self,
                 wav_root='../Dataset/wav',
                 wav_record='../Dataset/wav.tfrecord',
                 wav_transform=None,
                 loader=default_loader):

        if os.path.exists(wav_record):
            print("File already existed! Will not be created.")
        data_root = pathlib.Path(wav_root)

        all_wav_path = list(data_root.glob('*/*/*.wav'))
        self.all_wav_list = [str(path) for path in all_wav_path]

        spk_label = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

        # Add index for all speakers label
        label_to_index = dict((name, index) for index, name in enumerate(spk_label))

        self.all_wav_labels = [label_to_index[pathlib.Path(path).parent.parent.name]
                          for path in all_wav_path]

        # wav_spec_list = np.zeros([len(all_wav_labels), 513, 300])

        # with open(txt_path, 'r') as f:
        #     lines = f.readlines()
        #     self.img_list = [
        #         os.path.join(img_path, i.split()[0]) for i in lines
        #     ]
        #     self.label_list = [i.split()[1] for i in lines]
        self.spk_label = spk_label
        self.wav_transform = wav_transform
        self.loader = loader

    def __getitem__(self, index):
        wav_path = self.all_wav_list[index]
        label = self.all_wav_labels[index]
        wav = self.loader(wav_path)

        if self.wav_transform is not None:
            wav = self.wav_transform(wav)
        return wav, label

    def __len__(self):
        return len(self.all_wav_labels)


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    wav, label = zip(*batch)
    pad_label = []
    lens = []
    max_len = len(label[0])
    for i in range(len(label)):
        temp_label = [0] * max_len
        temp_label[:len(label[i])] = label[i]
        pad_label.append(temp_label)
        lens.append(len(label[i]))
    return wav, pad_label, lens
