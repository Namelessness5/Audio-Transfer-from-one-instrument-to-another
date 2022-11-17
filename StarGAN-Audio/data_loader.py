from torch.utils import data
import torch
import os
import random
import glob
from os.path import join, basename, dirname, split
import numpy as np
import os

# Below is the accent info for the used 10 speakers.
path = "/project/duandingcheng/IRMAS-TrainingData-trans"
speakers = os.listdir(path)
for item in speakers:
    if "." in item:
        speakers.remove(item)

# spk2acc = {'262': 'Edinburgh', #F
#            '272': 'Edinburgh', #M
#            '229': 'SouthEngland', #F
#            '232': 'SouthEngland', #M
#            '292': 'NorthernIrishBelfast', #M
#            '293': 'NorthernIrishBelfast', #F
#            '360': 'AmericanNewJersey', #M
#            '361': 'AmericanNewJersey', #F
#            '248': 'India', #F
#            '251': 'India'} #M
min_length = 256   # Since we slice 256 frames from each utterance when training.
# Build a dict useful when we want to get one-hot representation of speakers.
# speakers = ['p262', 'p272', 'p229', 'p232', 'p292', 'p293', 'p360', 'p361', 'p248', 'p251']
spk2idx = dict(zip(speakers, range(len(speakers))))

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    From Keras np_utils
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class MyDataset(data.Dataset):
    """Dataset for MCEP features and speaker labels."""
    def __init__(self, data_dir):
        mc_files = glob.glob(join(data_dir, '*.npy'))
        Remove = []
        for item in mc_files:
            if not item.split("[")[1][0:3] in speakers:
                Remove.append(item)
            if not os.path.exists(item):
                Remove.append(item)
        for item in Remove:
            mc_files.remove(item)
        self.mc_files = mc_files
        # mc_files = [i for i in mc_files if basename(i)[:4] in speakers]
        self.num_files = len(self.mc_files)
        print("\t Number of training samples: ", self.num_files)


    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        filename = self.mc_files[index]
        for item in speakers:
            if item in filename:
                spk = item
        spk_idx = spk2idx[spk]
        mc = np.load(filename)
        # to one-hot
        spk_cat = np.squeeze(to_categorical([spk_idx], num_classes=len(speakers)))

        return torch.FloatTensor(mc), torch.LongTensor([spk_idx]).squeeze_(), torch.FloatTensor(spk_cat)
        

class TestDataset(object):
    """Dataset for testing."""
    def __init__(self, data_dir, wav_dir, src_spk='p262', trg_spk='p272'):
        self.src_spk = src_spk
        self.trg_spk = trg_spk
        
        mc_files = sorted(glob.glob(join(data_dir, '*.npy')))
        Remove = []
        for item in mc_files:
            if not item.split("[")[1][0:3] in speakers:
                Remove.append(item)
            if not os.path.exists(item):
                Remove.append(item)
        for item in Remove:
            mc_files.remove(item)
        self.mc_files = mc_files

        self.src_wav_dir = f'{wav_dir}/{src_spk}'
        self.spk_idx = spk2idx[trg_spk]
        # self.spk_idx = trg_spk
        spk_cat = to_categorical([self.spk_idx], num_classes=len(speakers))
        self.spk_c_trg = spk_cat

    def get_batch_test_data(self, batch_size=8):
        batch_data = []
        for i in range(batch_size):
            mcfile = self.mc_files[i]
            filename = basename(mcfile).split('-')[-1]
            wavfile_path = self.src_wav_dir.split(self.src_spk)[0] + mcfile.split("[")[1][0:3] + "/" + filename.replace('npy', 'wav')[1:]
            batch_data.append(wavfile_path)
        return batch_data       

def get_loader(data_dir, batch_size=32, mode='train', num_workers=1):
    dataset = MyDataset(data_dir)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers,
                                  drop_last=True)
    return data_loader


if __name__ == '__main__':
    loader = get_loader('./data/mc/train')
    data_iter = iter(loader)
    for i in range(10):
        mc, spk_idx, acc_idx, spk_acc_cat = next(data_iter)
        print('-'*50)
        print(mc.size())
        print(spk_idx.size())
        print(acc_idx.size())
        print(spk_acc_cat.size())
        print(spk_idx.squeeze_())
        print(spk_acc_cat)
        print('-'*50)







