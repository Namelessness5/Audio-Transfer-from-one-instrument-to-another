import librosa
import numpy as np
import pyworld
import soundfile as sf



path = "CQTpiano.wav"


def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):

    # Logarithm Gaussian normalization for Pitch Conversions
    f0_converted = np.exp((np.ma.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)
    return f0_converted


def load_wav(wav_file, sr):
    wav, _ = librosa.load(wav_file, sr=sr, mono=True)
    return wav

def world_decompose(wav, fs, frame_period = 5.0):
    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor=71.0, f0_ceil=800.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    return f0, timeaxis, sp, ap

def world_encode_spectral_envelop(sp, fs, dim=36):
    # Get Mel-cepstral coefficients (MCEPs)
    #sp = sp.astype(np.float64)
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)
    return coded_sp

def world_decode_spectral_envelop(coded_sp, fs):
    # Decode Mel-cepstral to sp
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)
    return decoded_sp

def world_encode_wav(wav_file, fs, frame_period=5.0, coded_dim=36):
    wav = load_wav(wav_file, sr=fs)
    f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=fs, frame_period=frame_period)
    coded_sp = world_encode_spectral_envelop(sp = sp, fs = fs, dim = coded_dim)
    return f0, timeaxis, sp, ap, coded_sp


def wav_padding(wav, sr, frame_period, multiple = 4):

    assert wav.ndim == 1
    num_frames = len(wav)
    num_frames_padded = int((np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) + 1) / multiple + 1) * multiple - 1) * (sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right), 'constant', constant_values = 0)

    return wav_padded

def logf0_statistics(f0s):
    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()

    return log_f0s_mean, log_f0s_std


def world_speech_synthesis(f0, coded_sp, ap, fs, frame_period):
    decoded_sp = world_decode_spectral_envelop(coded_sp, fs)
    # TODO
    min_len = min([len(f0), len(coded_sp), len(ap)])
    f0 = f0[:min_len]
    coded_sp = coded_sp[:min_len]
    ap = ap[:min_len]
    wav = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)
    # Librosa could not save wav if not doing so
    wav = wav.astype(np.float32)
    return wav


def trans(path):
    frame_period = 5

    wav, sr = librosa.load(path, sr=16000, mono=True)

    wav = np.float64(wav)
    _f0, t = pyworld.harvest(wav, sr)  # 提取基频
    f0 = pyworld.stonemask(wav, _f0, t, sr)  # 修改基频
    sp = pyworld.cheaptrick(wav, f0, t, sr)  # 提取频谱包络
    ap = pyworld.d4c(wav, f0, t, sr)  # 提取非周期性指数
    wav_transformed = pyworld.synthesize(f0, sp, ap, sr)

    # wav = wav_padding(wav, sr=sr, frame_period=5, multiple=4)  # TODO
    # f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sr, frame_period=frame_period)
    # # decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
    # wav_transformed = world_speech_synthesis(f0=f0, coded_sp=sp, ap=ap, fs=sr, frame_period=frame_period)

    sf.write("Test.wav", wav_transformed, sr, "PCM_16")

if __name__ == "__main__":
    trans(path)