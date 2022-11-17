import librosa
import numpy as np
import soundfile as sf
import os
from utils import *


n_fft = 512


path = "data/notes15/"
tpath = "data/note15_t/"


# style1_path = "data/freqG/"
style1_path = "data/snG"
style2_path = "data/snP"
convert_path1 = style1_path + "c" + "/"
convert_path2 = style2_path + "c" + "/"
convert_npy1 = style1_path + "cn" + "/"
convert_npy2 = style2_path + "cn" + "/"
convert_cqt1 = style1_path + "cqt" + "/"
convert_cqt2 = style2_path + "cqt" + "/"
style1_path += "/"
style2_path += "/"

def trans(path=path):
    if not os.path.exists(tpath):
        os.makedirs(tpath)
    waves = os.listdir(path)
    for item in waves:
        wave = path + item
        freq = get_base_freq(wave)
        y, sr = librosa.load(wave)
        sf.write(tpath + "%.3f"%(freq) + ".wav", y, sr, "PCM_16")


def write_data():
    waves1 = os.listdir(style1_path)
    waves2 = os.listdir(style2_path)

    if not os.path.exists(convert_path1):
        os.makedirs(convert_path1)
    if not os.path.exists(convert_path2):
        os.makedirs(convert_path2)
    if not os.path.exists(convert_npy1):
        os.makedirs(convert_npy1)
    if not os.path.exists(convert_npy2):
        os.makedirs(convert_npy2)
    if not os.path.exists(convert_cqt1):
        os.makedirs(convert_cqt1)
    if not os.path.exists(convert_cqt2):
        os.makedirs(convert_cqt2)

    index = 0
    if len(waves1) > len(waves2):
        tw = waves2
        waves2 = waves1
        waves1 = tw
    for i in range(len(waves1)):
        y1, sr = librosa.load(style1_path + waves1[i])
        y2, sr = librosa.load(style2_path + waves2[i])
        e1 = np.sum(np.abs(y1))
        e2 = np.sum(np.abs(y2))
        if e1 > e2:
            y2 *= e1 / e2
        else:
            y1 *= e2 / e1

        sf.write(convert_path1 + "{}.wav".format(index), y1, sr, "PCM_16")
        sf.write(convert_path2 + "{}.wav".format(index), y2, sr, "PCM_16")

        y1s = librosa.stft(y1, n_fft=n_fft)
        y2s = librosa.stft(y2, n_fft=n_fft)
        y1s = np.abs(y1s.transpose(1, 0))
        y2s = np.abs(y2s.transpose(1, 0))
        np.save(convert_npy1 + "{}.npy".format(index), y1s, allow_pickle=False)
        np.save(convert_npy2 + "{}.npy".format(index), y2s, allow_pickle=False)

        y1c = librosa.cqt(y1, hop_length=64)
        y2c = librosa.cqt(y2, hop_length=64)
        y1c = np.abs(y1c.transpose(1, 0))
        y2c = np.abs(y2c.transpose(1, 0))
        np.save(convert_cqt1 + "{}.npy".format(index), y1c, allow_pickle=False)
        np.save(convert_cqt2 + "{}.npy".format(index), y2c, allow_pickle=False)
        index += 1



def write_data_freq():
    freq1, wav1 = get_style(style1_path)
    freq2, wav2 = get_style(style2_path)

    if not os.path.exists(convert_path1):
        os.makedirs(convert_path1)
    if not os.path.exists(convert_path2):
        os.makedirs(convert_path2)
    if not os.path.exists(convert_npy1):
        os.makedirs(convert_npy1)
    if not os.path.exists(convert_npy2):
        os.makedirs(convert_npy2)
    if not os.path.exists(convert_cqt1):
        os.makedirs(convert_cqt1)
    if not os.path.exists(convert_cqt2):
        os.makedirs(convert_cqt2)

    index = 0
    if len(freq1) > len(freq2):
        tf = freq2
        tw = wav2
        freq2 = freq1
        wav2 = wav1
        freq1 = tf
        wav1 = tw

    for i in range(len(wav1)):
        freq_index = findnearest(freq2, freq1[i])
        freq = freq2[freq_index]
        if np.abs(freq - freq1[i]) <= 10:
            print(freq)
            y1, sr = librosa.load(wav1[i])
            y2, sr = librosa.load(wav2[freq_index])
            y1, _ = librosa.effects.trim(y1, top_db=50, frame_length=256, hop_length=64)
            y2, _ = librosa.effects.trim(y2, top_db=50, frame_length=256, hop_length=64)

            if len(y2) < len(y1):
                rate = len(y2) / len(y1)
                y2 = librosa.effects.time_stretch(y2, rate)  # 时间缩放
            else:
                rate = len(y1) / len(y2)
                y1 = librosa.effects.time_stretch(y1, rate)  # 时间缩放

            if len(y1) > len(y2):
                y2 = np.pad(y2, (0, len(y1) - len(y2)), "constant")
            else:
                y1 = np.pad(y1, (0, len(y2) - len(y1)), "constant")

            e1 = np.sum(np.abs(y1))
            e2 = np.sum(np.abs(y2))
            if e1 > e2:
                y2 *= e1 / e2
            else:
                y1 *= e2 / e1

            y1 = y1[0:len(y1)//2]
            y2 = y2[0:len(y2)//2]
            sf.write(convert_path1 + "{}.wav".format(index), y1, sr, "PCM_16")
            sf.write(convert_path2 + "{}.wav".format(index), y2, sr, "PCM_16")

            y1s = librosa.stft(y1, n_fft=n_fft)
            y2s = librosa.stft(y2, n_fft=n_fft)
            y1s = np.abs(y1s.transpose(1, 0))
            y2s = np.abs(y2s.transpose(1, 0))
            np.save(convert_npy1 + "{}.npy".format(index), y1s, allow_pickle=False)
            np.save(convert_npy2 + "{}.npy".format(index), y2s, allow_pickle=False)

            y1c = librosa.cqt(y1, hop_length=64)
            y2c = librosa.cqt(y2, hop_length=64)
            y1c = np.abs(y1c.transpose(1, 0))
            y2c = np.abs(y2c.transpose(1, 0))
            np.save(convert_cqt1 + "{}.npy".format(index), y1c, allow_pickle=False)
            np.save(convert_cqt2 + "{}.npy".format(index), y2c, allow_pickle=False)
            index += 1


if __name__ == "__main__":
    # write_data_freq()
    write_data()
    # trans()

