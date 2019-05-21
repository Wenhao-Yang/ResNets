#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: GenrateSpectrum.py
@Time: 2019/5/11 16:19
@Overview: This recipe is trying to calculate the spectrogram for all the datasets. And the parameters for calculation is the same as the paper mentioned:

 All audio is first converted to single-channel, 16-bit streams at a 16kHz sampling rate for consistency. Spectrograms are then generated in a sliding window fashion using a hamming window of width 25ms, step 10ms and 1024-point FFT. This gives spectrograms of size 512 x 300 for 3 seconds of speech. Mean and variance normalisation is performed on every frequency bin of the spectrum. This normalisation is crucial, leading to an almost 10% increase in classification accuracy, as shown in Table 7. No other speech-specific preprocessing (e.g. silence removal, voice activity detection, or removal of unvoiced speech) is used.


"""
from scipy import signal
import matplotlib.pyplot as plt
from scipy.io import wavfile
import tensorflow


sample_rate, samples = wavfile.read('../Dataset/wav/id10001/1zcIwhmdeo4/00001.wav')
# samples = samples[:3*sample_rate]
print(samples.shape)
frequencies, times, spectrogram = signal.spectrogram(x=samples, fs=sample_rate, window=signal.hamming(25*16), noverlap=15*16, nfft=1024)
frequencies, times, spectrogram = frequencies[:], times[:300], spectrogram[:,:300]
plt.pcolormesh(times, frequencies, spectrogram)
# plt.axis('off')
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
#plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
plt.savefig('../Dataset/wav/id10001/1zcIwhmdeo4/00001.png', format='png',  dpi=300)
print(spectrogram.shape)
#plt.imshow(spectrogram)
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()
