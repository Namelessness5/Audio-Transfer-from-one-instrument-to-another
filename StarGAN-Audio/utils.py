import librosa
import numpy as np
import pyworld
import soundfile as sf


# split the audio into n * part(second)
def Split_Convert(path, rate=22050, part=1, Method="CQT", length=128):
    y, sr = librosa.load(path, sr=rate, mono=True)
    audios = []
    result = []

    # if the last part is shorter than half of the unit part, abandon it.
    time = len(y) / sr / part
    if (int(time) + 0.5) > time:
        for j in range(int(time)):
            audios.append(y[j * sr * part:(j + 1) * sr * part - 1])
    else:
        diff = (time + 1) * sr - len(y)
        y = np.pad(y, (0, diff), mode="constant")
        for j in range(int(time) + 1):
            audios.append(y[j * sr * part:(j + 1) * sr * part - 1])

    # Run the Convert
    if Method == "CQT":
        for i in range(len(audios)):
            TransAudio = np.abs(librosa.cqt(audios[i], hop_length=length))
            result.append(TransAudio)
    return result

    # for test
    # rec = librosa.icqt(result[0], hop_length=length)
    # for i in range(1, len(result)):
    #     temp = librosa.icqt(result[i], hop_length=length)
    #     rec = np.concatenate((rec, temp), axis=0)
    # sf.write("Test.wav", rec, sr, "PCM_16")


def Split_Convert_A(path, rate=22050, part=1, Method="CQT", length=128):
    y, sr = librosa.load(path, sr=rate, mono=True)
    audios = []
    result = []
    angle = []

    # if the last part is shorter than half of the unit part, abandon it.
    time = len(y) / sr / part
    if (int(time) + 0.5) > time:
        for j in range(int(time)):
            audios.append(y[j * sr * part:(j + 1) * sr * part - 1])
    else:
        diff = (time + 1) * sr - len(y)
        y = np.pad(y, (0, diff), mode="constant")
        for j in range(int(time) + 1):
            audios.append(y[j * sr * part:(j + 1) * sr * part - 1])

    # Run the Convert
    if Method == "CQT":
        for i in range(len(audios)):
            TransAudio = librosa.cqt(audios[i], hop_length=length)
            angle.append(np.angle(TransAudio))
            result.append(np.abs(TransAudio))
    return result, angle

