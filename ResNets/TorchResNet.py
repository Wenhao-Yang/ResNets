#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: TorchResNet.py
@Time: 19-5-23 下午5:14
@Overview:
"""
import torch
import torchvision
import torchvision.transforms as transforms
from Data_Process import Voxceleb
import matplotlib as plt
import numpy as np

from PIL import Image
img = Image.open('../Dataset/wav/id10001/1zcIwhmdeo4/00001.png')
img_arr = np.array(img)
img_t = (transforms.ToTensor())(img)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = Voxceleb.voxceleb(wav_transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)


spks = trainset.spk_label

def spectshow(wav_spect):
    wav_spect = wav_spect / 2 + 0.5     # unnormalize
    npimg = wav_spect.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
spectshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % spks[labels[j]] for j in range(4)))