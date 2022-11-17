import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft, fft, ifft, fftfreq
import soundfile as sf
import os


path1 = "data/cn/"
path2 = "data/cP/"
path = "audio.wav"
high_freq = 4000


def compare(path1=path1, path2=path2):
    paths = os.listdir(path1)
    plt.figure()
    block = 128
    for wave in paths:
        plt.clf()
        plt.subplot(211)
        audio, sr = librosa.load(path1 + wave)
        # plt.plot(np.linspace(0, len(audio), len(audio)), audio)
        plt.specgram(audio, Fs=sr, noverlap=int(block / 2), NFFT=block)
        plt.subplot(212)
        audio, sr = librosa.load(path2 + wave)
        # plt.plot(np.linspace(0, len(audio), len(audio)), audio)
        plt.specgram(audio, Fs=sr, noverlap=int(block / 2), NFFT=block)
        plt.show()


def s(path):
    block = 128
    audio, sr = librosa.load(path, None)
    # plt.plot(np.linspace(0, len(audio), len(audio)), audio)
    plt.specgram(audio, Fs=sr, noverlap=int(block/2), NFFT=block)
    plt.show()


def temp1(path=path):
    y, sr = librosa.load(path)
    x = np.linspace(0, sr, len(y))
    ft = np.abs(fft(y))

    plt.figure(figsize=(10, 8))
    plt.subplot(111)
    plt.plot(x, ft, linewidth=1)
    plt.show()


def temp2(path):
    y, sr = librosa.load(path)
    stft = librosa.stft(y, n_fft=2048, win_length=1024)
    print("a")


def filt(path):
    audio, sr = librosa.load(path, sr=24000)
    for i in range(600):
        audio[i] = 0

    duration = len(audio)//sr
    for i in range(800):
        for j in range(duration):
            if j * sr + i < len(audio):
                audio[j * sr + i - 200] = 0
    sf.write("Trans" + path, audio, sr, "PCM_16")


if __name__ == "__main__":
    # temp1(path)
    # temp2(path)
    # s(path)
    # filt(path)
    compare()
