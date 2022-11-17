import numpy as np
import os, sys
import argparse
import pyworld
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import utils
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import glob
from os.path import join, basename
import subprocess
import librosa
import soundfile as sf


def split_data(paths):
    indices = np.arange(len(paths))
    test_size = 0.1
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=1234)
    train_paths = list(np.array(paths)[train_indices])
    test_paths = list(np.array(paths)[test_indices])
    return train_paths, test_paths


def get_spk_world_feats(spk_fold_path, mc_dir_train, mc_dir_test, sr):
    paths = glob.glob(join(spk_fold_path, '*.wav'))
    spk_name = basename(spk_fold_path)
    train_paths, test_paths = split_data(paths)

    for wav_file in tqdm(train_paths):
        wav_nam = basename(wav_file)
        results = utils.Split_Convert(wav_file, rate=sr)
        for i in range(len(results)):
            np.save(mc_dir_train + "/" + str(i) + wav_nam.replace('.wav', '.npy'), results[i], allow_pickle=False)

    for wav_file in tqdm(test_paths):
        wav_nam = basename(wav_file)
        results = utils.Split_Convert(wav_file, rate=sr)
        for i in range(len(results)):
            np.save(mc_dir_test + "/" + str(i) + wav_nam.replace('.wav', '.npy'), results[i], allow_pickle=False)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    sample_rate_default = 24000
    origin_wavpath_default = "F://IRMAS-TrainingData"
    target_wavpath_default = "F://IRMAS-TrainingData-trans"
    mc_dir_train_default = './data/mc/train'
    mc_dir_test_default = './data/mc/test'
    Method_default = "CQT"

    parser.add_argument("--sample_rate", type=int, default=sample_rate_default, help="Sample rate.")
    parser.add_argument("--origin_wavpath", type=str, default=origin_wavpath_default, help="The original wav path to "
                                                                                           "resample.")
    parser.add_argument("--target_wavpath", type=str, default=target_wavpath_default, help="The original wav path to "
                                                                                           "resample.")
    parser.add_argument("--mc_dir_train", type=str, default=mc_dir_train_default, help="The directory to store the "
                                                                                       "training features.")
    parser.add_argument("--mc_dir_test", type=str, default=mc_dir_test_default, help="The directory to store the "
                                                                                     "testing features.")
    parser.add_argument("--num_workers", type=int, default=None, help="The number of cpus to use.")

    argv = parser.parse_args()

    sample_rate = argv.sample_rate
    origin_wavpath = argv.origin_wavpath
    target_wavpath = argv.target_wavpath
    mc_dir_train = argv.mc_dir_train
    mc_dir_test = argv.mc_dir_test
    method = Method_default
    num_workers = 1  # argv.num_workers if argv.num_workers is not None else cpu_count()

    speaker_used = os.listdir(origin_wavpath_default)
    for item in speaker_used:
        if "." in item:
            speaker_used.remove(item)

    os.makedirs(mc_dir_train, exist_ok=True)
    os.makedirs(mc_dir_test, exist_ok=True)

    # num_workers = cpu_count()
    # print("number of workers: ", num_workers)
    # executor = ProcessPoolExecutor(max_workers=num_workers)

    work_dir = target_wavpath
    # spk_folders = os.listdir(work_dir)
    # print("processing {} speaker folders".format(len(spk_folders)))
    # print(spk_folders)

    futures = []
    for spk in speaker_used:
        spk_path = os.path.join(work_dir, spk)
        get_spk_world_feats(spk_path, mc_dir_train, mc_dir_test, sample_rate)

    sys.exit(0)
