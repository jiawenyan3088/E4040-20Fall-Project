import numpy as np
import soundfile as sf
from scipy.fftpack import dct
import math
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Lambda

def melspectrogram(x_cpu):
        pre_emph = 0.97
        frame_size = 0.025
        frame_stride = 0.01
        sr = 16000
        NFFT = 512
        nfilt = 80
        #Pre Emphasis
        x_cpu_emph = np.append(x_cpu[0], x_cpu[1:] - pre_emph * x_cpu[:-1])
        #Framing
        frame_length, frame_step = frame_size * sr, frame_stride * sr
        signal_length = len(x_cpu_emph)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(x_cpu_emph, z)
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        #Window
        frames *= np.hamming(frame_length)
        #FFT and Power Spectrum
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
        pow_frames = np.float32(pow_frames)
        #Filter Banks
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700)) 
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
        hz_points = (700 * (10**(mel_points / 2595) - 1)) 
        bin = np.floor((NFFT + 1) * hz_points / sr)
        bin = np.float32(bin)
        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        count = 0
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
                count += 1
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

        fbank_T_cpu = fbank.T
        fbank_T_cpu = np.float32(fbank_T_cpu)   
        filter_banks = np.dot(pow_frames, fbank_T_cpu)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)  # dB
        #Normalization
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
        return  filter_banks
    
Output = Lambda(melspectrogram)(x_cpu,0.97,0.025,0.01,16000,512,80)

#Parameters:
#pre_emph = 0.97
#frame_size = 0.025
#frame_stride = 0.01
#NFFT = 512 
#nfilt = 80

#Usage:
#function = melspectrogram(data,pre_emph,frame_size,frame_stride,sr,NFFT,nfilt)
#where 'data' should be one dimensional 1 by N numpy array (converted from audio sample to numpy array)
#plt.figure(figsize=(17,6))
#plt.pcolormesh(function)
#plt.show
#Then the figure should appear

#Caution:
#If an audio file is stereo, it would be a 2 by N numpy array. Check the shape before use it.

#Output:
#Spectrogram of that certain audio file.
  

    
