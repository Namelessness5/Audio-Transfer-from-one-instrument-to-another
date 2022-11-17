import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft, fft, ifft, fftfreq
import soundfile as sf


path = "result.wav"
high_freq = 4000


def s(path=path):
    block = 128
    audio, sr = librosa.load(path, None)
    plt.plot(np.linspace(0, len(audio), len(audio)), audio)
    # plt.specgram(audio, Fs=sr, noverlap=int(block/2), NFFT=block)
    plt.show()


def temp1(path):
    path = "stft500_cpiano2.wav_rephase_guitar.wav_sw4000.wav"
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
    filt(path)
