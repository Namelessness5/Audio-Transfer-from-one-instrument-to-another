import numpy as np
import torch
from model import transfer
import librosa
import soundfile as sf

# You don't need to input .wav or .pth
audio_name = "For~2"
# model_path = "model/1-1modelG-P_conv_loss_1.9"
model_path = "model/bestmodelP-n_conv_loss_n"
origin_path = "org_audio/"
result_p = "results/"

n_fft = 512
method = "conv"
conv_dim = 128
model_depth = 8                  # for conv: depth should be greater than 2
kernel_size = 7
bias = False

if method == "conv":
    result_path = result_p + audio_name + "_" + method + "_" + "loss" + model_path.split("loss")[-1]
else:
    result_path = result_p + audio_name + "_" + method
origin_audio_path = origin_path + audio_name + ".wav"

model = transfer(n_fft // 2 + 1, model_depth, method, conv_dim, kernel_size=kernel_size, bias=bias)
dict = torch.load(model_path + ".pth", map_location=torch.device('cpu'))
model.load_state_dict(dict)


if method == "conv":
    cqt = True
else:
    cqt = False


def cqt_convert():
    y, sr = librosa.load(origin_audio_path)
    # y = y[3*len(y)//4: 4*len(y)//4]
    ft = librosa.cqt(y, hop_length=64)
    angle = np.angle(ft)
    ft = ft.transpose(1, 0)
    ft = np.abs(ft)
    if method == "conv":
        ft = ft.reshape((ft.shape[0], 1, ft.shape[1]))

    ft = torch.FloatTensor(ft)
    out = model(ft)
    out = out.detach().cpu().numpy()

    if method == "conv":
        out = out.reshape((out.shape[0], out.shape[2]))
    out = out.transpose(1, 0)

    out = out * np.exp(1j * angle)
    out = librosa.icqt(out, hop_length=64)

    sf.write(result_path + ".wav", out, sr, "PCM_16")


def stft_convert():
    y, sr = librosa.load(origin_audio_path)
    ft = librosa.stft(y, n_fft=n_fft)
    angle = np.angle(ft)
    ft = ft.transpose(1, 0)
    ft = np.abs(ft)
    if method == "conv":
        ft = ft.reshape((ft.shape[0], 1, ft.shape[1]))

    ft = torch.FloatTensor(ft)
    out = model(ft)
    out = out.detach().cpu().numpy()

    if method == "conv":
        out = out.reshape((out.shape[0], out.shape[2]))
    out = out.transpose(1, 0)

    # p = 2 * np.pi * np.random.random_sample(out.shape) - np.pi
    # for i in range(100):
    #     S = out * np.exp(1j * p)
    #     x = librosa.istft(S)
    #     p = np.angle(librosa.stft(x, n_fft=n_fft))

    out = out * np.exp(1j * angle)
    out = librosa.istft(out)

    sf.write(result_path + ".wav", out, sr, "PCM_16")
    # sf.write(result_path + " random phase.wav", x, sr, "PCM_16")


if __name__ == "__main__":
    if cqt:
        cqt_convert()
    else:
        stft_convert()
