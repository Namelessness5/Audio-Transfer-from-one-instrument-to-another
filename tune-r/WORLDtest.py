import pyworld as pw
import librosa
import soundfile as sf
import numpy as np

path = "for~1.wav"

data, fs = librosa.load(path)
data = np.float64(data)
f0, t = pw.dio(data, fs)  #  提取基频
# f0 = pw.stonemask(data, _f0, t, fs)  # 修改基频
sp = pw.cheaptrick(data, f0, t, fs)  # 提取频谱包络
ap = pw.d4c(data, f0, t, fs)  # 提取非周期性指数

print("a")