import copy

import numpy
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.fftpack import hilbert, fft, ifft
import soundfile as sf
import pyworld as pw


def devidation(audio_path, wave_rate=2000):
    audio, sr = librosa.load(audio_path,None)
    audio_time=librosa.get_duration(filename=audio_path)
    i = 0
    wave = []
    wave_index = []
    while i < sr * audio_time:
        MAX = max(audio[i:i + wave_rate])
        wave.append(MAX)
        wave_index.append(i + numpy.argwhere(audio[i:i + wave_rate] == MAX))
        i += wave_rate
    points = []
    i = 2
    flag = False
    while i < len(wave):
        if (wave[i - 2] < wave[i] - 0.1) & (not flag):
            points.append(wave_index[i])
            flag = True
        if wave[i - 2] > wave[i]:
            flag = False
        i += 1
    return points

def vol_D(audio_path, wave_rate=2000, s_rate=100, rate=0.5):
    audio , sr = librosa.load(audio_path, None)
    points = devidation(audio_path)
    endpoints = []
    parts = []
    audio = np.array(audio)

    for i in range(len(points)):
        j = 0
        tmpmax = max(audio[int(points[i]) : int(points[i] + wave_rate)])
        while max(audio[int(points[i] + j) : int(points[i] + j + s_rate)]) > rate * tmpmax or max(audio[int(points[i] + j + s_rate) : int(points[i] + j + 2 * s_rate)]) > rate * tmpmax:
            j += s_rate
        endpoints.append(points[i] + j)

    for i in range(len(points)):
        temp = []
        a = numpy.argwhere(abs(audio[int(points[i]):]) < 0.005)
        b = numpy.argwhere(abs(audio[int(endpoints[i]):]) < 0.005)
        temp.append(int(a[0][0]) + int(points[i]))
        temp.append(int(b[0][0]) + int(endpoints[i]))
        parts.append(temp)
    return parts


def lengthen(audio, times, vol=1):
    for i in range(int(times)):
        audio = np.concatenate((audio, audio))
    audio *= vol
    return audio

if __name__ == "__main__":
    audio_path = "for~1.wav"
    save_path = "outputG/"
    audio, sr = librosa.load(audio_path, None)

    '''
    parts = vol_D(audio_path)
    for i in range(len(parts)):
        #audio[int(parts[i][0]): int(parts[i][1])] = 1
        tmp = lengthen(audio[int(parts[i][0]) : int(parts[i][1])], times=5, vol=2)
        sf.write(save_path + str(i) + ".wav", tmp, sr, 'PCM_16')'''
    #plt.plot(audio)
    #plt.show()

    de = devidation(audio_path)

    for i in range(len(de) - 1):
        tmp = audio[int(de[i]) : int(de[i + 1]) - 4000]

        # tmp = np.float64(tmp)
        # bf, t = pw.dio(tmp, sr)
        # temp = []
        # for j in range(len(bf)):
        #     if bf[j] > 50:
        #         temp.append(bf[j])
        # avg = np.median(temp)

        ft = abs(fft(tmp))
        #MAX = max(ft[0:int(2000 / sr * len(tmp))])
        MAX = max(ft[0 : len(ft)//2])
        index = np.argwhere(ft == MAX)
        print(str(index[0] / len(ft) * sr) + "Hz")

        sf.write(save_path + str(i) + ".wav", tmp, sr, 'PCM_16')
    tmp = audio[int(de[-1]):]

    ft = abs(fft(tmp))
    #MAX = max(ft[0:int(2000 / sr * len(tmp))])
    MAX = max(ft[0 : len(ft)//2])
    index = np.argwhere(ft == MAX)
    print(str(index[0] / len(ft) * sr) + "Hz")

    # tmp = np.float64(tmp)
    # bf, t = pw.dio(tmp, sr)
    # temp = []
    # for j in range(len(bf)):
    #     if bf[j] > 50:
    #         temp.append(bf[j])
    # avg = np.median(temp)

    sf.write(save_path + str(i) + ".wav", tmp, sr, 'PCM_16')




'''
y = abs(audio)
for i in range(len(points)):
    for j in range(10):
        y[points[i] + j] = 1'''

#pic = fft(audio)
#y=abs(pic)
#test = hilbert(audio)
#out = np.sqrt(audio**2 + test**2)

'''print(y[275])
print(y[550])
print(y[824])
print(y[1099])
print(y[1649])
print(y[1924])
print(y[2198])
print(y[2455])
n=0
max=0
for i in range(20000):
    if y[i]>max:
        n=i
        max=y[i]
print(n)
for i in range(10):
    print(y[(i+1)*n])'''

'''for i in range(int(20*audio_time)):
    pic=fft(audio[0+2205*i:2205*(i+1)])
    y=abs(pic)
    plt.plot(y)
    plt.show()'''
'''
plt.plot(y)
plt.show()
print("Hello!")
'''
