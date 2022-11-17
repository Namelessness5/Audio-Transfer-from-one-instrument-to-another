import pyworld as pw
import librosa
import soundfile as sf
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy


low_freq = 20
test_path = "outputG/2.wav"
basic_path = "piano/"
freq_path = "freqP/"
origin_path = "audio.wav"
WINDOW = "rect"
# origin_path = "CQTpiano.wav"
# origin_path = "for~1.wav"


def ZeroCR(waveData, frameSize, overLap):
    wlen = len(waveData)
    step = frameSize - overLap
    frameNum = int(wlen/step)
    zcr = np.zeros((len(waveData)))
    for i in range(frameNum):
        curFrame = waveData[np.arange(i*step, min(i*step+frameSize, wlen))]

        #To avoid DC bias, usually we need to perform mean subtraction on each frame
        curFrame = curFrame - np.mean(curFrame) # zero-justified
        framezcr = sum(curFrame[0:-1]*curFrame[1::] < 0)
        for j in range(frameSize):
            zcr[i * frameSize + j] = framezcr
    return zcr


def cut_signal(sig, nw=512, inc=128):
    signal_length = len(sig)  # length of signal
    if signal_length <= nw:  # if length < length of window set frame number = 1
        nf = 1  # number of frame
    else:
        nf = int(np.ceil((1.0 * signal_length - nw + inc) / inc))  # desert the last frame
    pad_length = int((nf - 1) * inc + nw)  # the sum of all the frames
    pad_signal = np.pad(sig, (0, pad_length - signal_length), 'constant')  # padding

    indices = np.tile(np.arange(0, nw), (nf, 1)) \
        + np.tile(np.arange(0, nf * inc, inc), (nw, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = pad_signal[indices]
    win = np.tile(np.hanning(nw), (nf, 1))
    return frames*win


def compute_zcr(cut_sig, low=0.02):
    # zero cross rate
    E = np.max(cut_sig)
    frames = cut_sig.shape[0]
    frame_size = cut_sig.shape[1]
    _zcr = np.zeros((frames, 1))

    for i in range(frames):
        frame = cut_sig[i]
        tmp1 = frame[:frame_size-1]
        tmp2 = frame[1:frame_size]
        signs = tmp1*tmp2.T < 0
        # energy limit
        diff = np.abs((tmp1-tmp2)) > low * E
        _zcr[i] = np.sum(signs.T * diff)
    return _zcr


def compute_amp(cut_sig):
    # energy in a short time
    return np.sum(np.abs(cut_sig), axis=1)


def maxfilt(audio, low_freq=low_freq):
    sr = 22050
    wave_rate = sr // low_freq + 1
    y = scipy.ndimage.maximum_filter1d(audio, size=wave_rate)
    return y


def get_Audio(path, org, window="rect"):
    t, sr = librosa.load(path)
    org_length = len(org)

    org, index1 = librosa.effects.trim(org, top_db=30, frame_length=256, hop_length=64) # cut the part of silence
    t, index2 = librosa.effects.trim(t, top_db=30, frame_length=256, hop_length=64)

    rate = len(t) / len(org)
    t = librosa.effects.time_stretch(t, rate)                                        # time stretch, to get the same length
    if len(t) > len(org):
        t = t[:len(org) - 1]
    elif len(t) < len(org):
        t = np.pad(t, (0, len(org) - len(t)), "constant")

    t = t * maxfilt(org) / maxfilt(t)
    t = np.pad(t, (0, org_length - len(t)), "constant")

    # using window to decrease the noise in each tune's border
    if window == "hanm":
        w = np.hamming(len(t))
        half = len(t)//2
        w = np.concatenate((np.ones(half), w[half:]), axis=0)
        t = t * w
    elif window == "hann":
        w = np.hanning(len(t))
        half = len(t)//2
        w = np.concatenate((np.ones(half), w[half:]), axis=0)
        t = t * w
    return t


def get_base_freq(path):
    y, sr = librosa.load(path)
    # piano(27.5 - 4186) guitar(82.4 - 1109)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, sr=sr, fmin=80, fmax=1000)
    freq = []
    for i in range(len(f0)):
        if voiced_flag[i] and voiced_probs[i] >= 0.9:
            freq.append(f0[i])
    if len(freq) == 0:
        return 0
    return np.mean(freq)


def get_base_freq1(y):
    sr = 22050
    # piano(27.5 - 4186) guitar(82.4 - 1109)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, sr=sr, fmin=80, fmax=1000)
    freq = []
    for i in range(len(f0)):
        if voiced_flag[i] and voiced_probs[i] >= 0.9:
            freq.append(f0[i])
    return np.mean(freq)


def findnearest(freqs, freq):
    freqs = np.array(freqs)
    freqs = np.abs(freqs - freq)
    index = np.argwhere(freqs == np.min(freqs))
    return index[0][0]


def get_style(fpath=freq_path):
    # the output is like [100, 20](Hz) [wave1.wav, wave2.wav]
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


def genStyle(fpath=freq_path, opath=basic_path):
    wavs = os.listdir(opath)
    for wave in wavs:
        y, sr = librosa.load(opath + wave)
        freq = get_base_freq1(y)
        if not freq == 0:
            sf.write(fpath + "%.3f"%(freq) + ".wav", y, sr, "PCM_16")


def devidation(audio_path):
    audio, sr = librosa.load(audio_path)
    filted = maxfilt(audio, low_freq)           # maxfilt
    diff2 = np.diff(filted, prepend=2)          # difference 2-order
    diff2[0] = 0
    diff2 = np.int32(diff2 >= 0.2 * max(diff2))
    e_feature = zcr_E_D(audio_path)             # energy feature
    result1 = e_feature * diff2
    d_points = []
    points = []
    for i in range(len(result1)):
        if result1[i] == 1:
            d_points.append(i)

    step = sr // 8
    avg = 0
    num = 0
    for i in range(len(d_points) - 1):
        if d_points[i + 1] - d_points[i] < step:
            avg += d_points[i]
            num += 1
        else:
            points.append(avg / num)
            avg = 0
            num = 0
        if i == len(d_points) - 2:
            points.append(avg / num)
            avg = 0
            num = 0

    # plt.figure()
    # plt.plot(result1)
    # plt.show()
    # print(len(points))
    return points


def convert(path="CQTpiano.wav"):
    audio, sr = librosa.load(path, None)
    de = devidation(path) # find the start of each tune
    result = np.array([0.0])
    freqs, waves = get_style()
    for i in range(len(de) - 1):
        tmp = audio[int(de[i]) : int(de[i + 1])]         # get each tune
        bf = get_base_freq1(tmp)                         # get base frequency
        index = findnearest(freqs, bf)                   # find the nearest frequency
        P = waves[index]
        temp = get_Audio(P, tmp, WINDOW)                         # time stretch and volume stretch
        result = np.concatenate((result, temp), axis=0)  # concatenate

    tmp = audio[int(de[-1]):]
    bf = get_base_freq1(tmp)
    P = waves[findnearest(freqs, bf)]
    temp = get_Audio(P, tmp)
    result = np.concatenate((result, temp), axis=0)

    sf.write(WINDOW + "_result.wav", result, sr, "PCM_16")


def zcr_E_D(path):
    framesize = 512
    overLap = 128

    y, sr = librosa.load(path)
    y_split = cut_signal(y, framesize, overLap)
    zcr_feature = compute_zcr(y_split)
    e_feature = compute_amp(y_split)
    x1 = np.linspace(0, len(y) / sr, len(y))
    x2 = np.linspace(0, len(y) / sr, len(zcr_feature))

    zcr_feature = np.diff(zcr_feature, prepend=2)
    e_feature = np.diff(e_feature, prepend=2)

    # zcr_feature = ZeroCR(y, frameSize=framesize, overLap=overLap)
    # plt.figure(figsize=(10, 6))
    # plt.subplot(311)
    # plt.plot(x1, y)
    # plt.subplot(312)
    # plt.plot(x2, zcr_feature)
    # plt.subplot(313)
    # plt.plot(x2, e_feature)
    # plt.show()

    result = np.zeros(len(y))
    for i in range(len(e_feature)):
        if e_feature[i] > 0.2 * np.max(e_feature):
            for j in range(2 * framesize):
                result[-framesize + j + int(i * (len(y) / len(e_feature)))] = 1
    return result


if __name__ == "__main__":
    # print(get_base_freq(test_path))
    # a = get_style()
    convert(origin_path)
    # devidation(origin_path)
    # genStyle()


