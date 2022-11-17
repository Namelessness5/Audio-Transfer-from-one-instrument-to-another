import librosa
import scipy
import numpy as np
import soundfile as sf


target_sr=16384
target_time=16


def resample1(y, sr, duartion, target_sr=8192, target_time=8):
    kind = "linear"
    new_sr = int(sr * (target_time * target_sr) / len(y))
    x = np.linspace(0, target_time, len(y))
    f = scipy.interpolate.interp1d(x, y, kind=kind)
    x_new = np.linspace(0, target_time, target_sr * target_time)
    result = f(x_new)
    return result, new_sr


def resample2(y, sr, target_sr=8192, target_time=8):
    new_sr = int(sr * (target_time * target_sr) / len(y))
    result = librosa.resample(y, sr, new_sr)
    result = np.pad(result, (0, target_sr * target_time - len(result)))
    return result, new_sr


if __name__ == "__main__":
    audio_path = "piano.wav"
    save_path = "resample_test.wav"
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y, sr)
    out, new_sr = resample1(y, sr, duration, target_sr, target_time)
    # out, new_sr = resample2(y, sr, target_sr, target_time)
    sf.write(save_path, out, new_sr, "PCM_16")