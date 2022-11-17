import librosa
import numpy as np
import soundfile as sf
import os


def get_style(fpath):
    wavs = os.listdir(fpath)
    freqs = []
    waves = []
    for item in wavs:
        path = fpath + item
        freq = float(item.split(".")[0])
        if not freq == 0 and not "nan" in item:
            freqs.append(freq)
            waves.append(path)
    return freqs, waves


def findnearest(freqs, freq):
    freqs = np.array(freqs)
    freqs = np.abs(freqs - freq)
    index = np.argwhere(freqs == np.min(freqs))
    return index[0][0]


def get_base_freq(path):
    y, sr = librosa.load(path)
    # piano(27.5 - 4186) guitar(82.4 - 1109)
    y, _ = librosa.effects.trim(y, top_db=40, frame_length=256, hop_length=64)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, sr=sr, fmin=80, fmax=1000)
    freq = []
    for i in range(len(f0)):
        if voiced_flag[i] and voiced_probs[i] >= 0.9:
            freq.append(f0[i])
    if len(freq) == 0:
        return 0
    return np.median(freq)


def get_base_freq1(y):
    sr = 22050
    # piano(27.5 - 4186) guitar(82.4 - 1109)
    y, _ = librosa.effects.trim(y, top_db=40, frame_length=256, hop_length=64)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, sr=sr, fmin=80, fmax=1000)
    freq = []
    for i in range(len(f0)):
        if voiced_flag[i] and voiced_probs[i] >= 0.9:
            freq.append(f0[i])
    return np.median(freq)