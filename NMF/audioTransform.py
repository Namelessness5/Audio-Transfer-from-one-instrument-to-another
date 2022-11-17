import librosa
import numpy as np
import soundfile as sf
from sklearn.decomposition import NMF


model = NMF(n_components=8, init='random', tol=5e-3)


if __name__ == "__main__":
    audio_path1 = "audios/for~2.wav"
    audio_path2 = "audios/guitar.wav"

    y1, sr1 = librosa.load(audio_path1)
    y2, sr2 = librosa.load(audio_path2)

    CQT_y1 = librosa.cqt(y=y1, sr=sr1, hop_length=64)
    y1_angle = np.angle(CQT_y1)
    abs_CQT_y1 = np.abs(CQT_y1)
    abs_CQT_y1 = abs_CQT_y1.transpose((1, 0))

    CQT_y2 = librosa.cqt(y=y2, sr=sr2, hop_length=64)
    abs_CQT_y2 = np.abs(CQT_y2)
    abs_CQT_y2 = abs_CQT_y2.transpose((1, 0))

    model.fit_transform(abs_CQT_y2)
    temp = model.transform(abs_CQT_y1)
    temp_CQT = model.inverse_transform(temp)
    temp_CQT = temp_CQT.transpose((1, 0))

    reconstruct = temp_CQT * np.exp(1j*y1_angle)

    result = librosa.icqt(C=reconstruct, sr=sr1, hop_length=64)
    sf.write("results/" + audio_path1.split(".")[0] + "_to_" + audio_path2, result, sr1, 'PCM_16')




