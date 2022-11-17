from torch.utils import data
import torch
import glob
from os.path import join
import numpy as np
import os


class MyDataset(data.Dataset):
    """Dataset for MCEP features and speaker labels."""
    def __init__(self, data_wave_path, data_label_path):
        wav_files = glob.glob(join(data_wave_path, '*.npy'))
        label_files = glob.glob(join(data_label_path, '*.npy'))


        self.wav_files = wav_files
        self.label_files = label_files
        self.num_files = len(self.wav_files)
        self.length = 0
        for i in range(self.num_files):
            self.length += np.load(wav_files[i]).shape[0]
        print("\t Number of training samples: ", self.num_files)


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        wavname = self.wav_files[index % self.num_files]
        labelname = self.label_files[index % self.num_files]

        wav = np.load(wavname)
        label = np.load(labelname)
        length = wav.shape[0]
        i = (index // self.num_files) % length

        return torch.FloatTensor(wav[i]), torch.FloatTensor(label[i])