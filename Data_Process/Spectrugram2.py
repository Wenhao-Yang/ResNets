#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: Spectrugram2.py
@Time: 2019/5/21 下午12:37
@Overview: plot the spectrugram using plt from matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import wave

# 读入音频。
path = "../Dataset/wav/id10001/1zcIwhmdeo4"
name = '00001.wav'
# 我音频的路径为E:\SpeechWarehouse\zmkm\zmkm0.wav
filename = os.path.join(path, name)

# 打开语音文件。
f = wave.open(filename, 'rb')
# 得到语音参数
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]

# 将字符串格式的数据转成int型
strData = f.readframes(nframes)
waveData = np.fromstring(strData, dtype=np.short)
# 归一化
waveData = waveData * 1.0 / max(abs(waveData))
# 将音频信号规整乘每行一路通道信号的格式，即该矩阵一行为一个通道的采样点，共nchannels行
waveData = np.reshape(waveData, [nframes, nchannels]).T  # .T 表示转置
f.close()  # 关闭文件

# 频谱
framelength = 0.025  # 帧长20~30ms
framesize = 1024  # 每帧点数 N = t*fs,通常情况下值为256或512,要与NFFT相等

sec3_data=waveData[0]
sec3_data=sec3_data[:3*framerate]
NFFT = framesize  # NFFT必须与时域的点数framsize相等，即不补零的FFT
overlapSize = 0.015 / 0.025* framesize  # 重叠部分采样点数overlapSize约为每帧点数的1/3~1/2
overlapSize = int(round(overlapSize))  # 取整
spectrum, freqs, ts, fig = plt.specgram(sec3_data,
                                        NFFT=NFFT,
                                        Fs=framerate,
                                        window=np.hamming(M=framesize),
                                        noverlap=overlapSize,
                                        mode='default',
                                        scale_by_freq=True,
                                        sides='default',
                                        scale='dB',
                                        xextent=None)  # 绘制频谱图
plt.ylabel('Frequency')
plt.xlabel('Time(s)')
plt.title('Spectrogram')
plt.show()

