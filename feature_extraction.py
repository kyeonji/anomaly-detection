"""
feature_extraction.py

Usage:
python feature_extraction.py KIND_DATA [--mode=MODE] [--num-workers=N]

- KIND_DATA can be 'train' or 'test'.
- MODE can be 'io', 'in', or 'out' (means what feature will be processed).
    Default is 'io'
- N can be an integer from 1 to cpu_count.
    Default is cpu_count - 1

"""
import multiprocessing as mp
import os
from argparse import ArgumentParser
from pathlib import Path

import librosa
import math
import numpy as np
from tqdm import tqdm

from hparams import hparams


def extract_feature(path_audio: Path):
    audio, _ = librosa.load(path_audio, sr=hparams.sample_rate)
    mel_spec = librosa.feature.melspectrogram(audio,
                                              sr=hparams.sample_rate,
                                              n_fft=hparams.fft_size,
                                              hop_length=hparams.hop_size,
                                              n_mels=hparams.n_mels,
                                              )
    mel_spec = mel_spec.astype(np.float32)
    mel_spec = mel_spec/1.48  # max magnitude of all data (train and test): 1.48
    mel_log = librosa.core.amplitude_to_db(mel_spec)

    mel_log = (mel_log+100)/(20 * math.log10(1.48) + 100)

    return mel_log


def main():

    pool = mp.Pool(num_workers)

    # make dataset : only for train data
    if kind_data == 'train':
        path_raw_data = hparams.path_raw_data[kind_data]
        raw_data = list(path_raw_data.glob('**/*.wav'))

        index = 0
        pbar = tqdm(raw_data, dynamic_ncols=True)
        for idx, path in enumerate(pbar):
            audio, _ = librosa.load(path, sr=hparams.sample_rate)
            for w in np.arange(1, 0, -0.1):  # weight for various mag level
                for start in range(0, len(audio), hparams.audio_length):
                    if start + hparams.audio_length > len(audio):
                        break
                    train_audio = w * audio[start:start + hparams.audio_length]  # trim audio 6sec
                    librosa.output.write_wav(hparams.path_dataset[kind_data]
                                             / f'audio_{index}.wav',
                                             train_audio,
                                             sr=hparams.sample_rate,)
                    index += 1

    path_data_folder = hparams.path_dataset[kind_data]
    file_list = list(path_data_folder.glob('**/*.wav'))

    results = dict()
    # get or make flie list and iterate
    pbar_apply = tqdm(file_list, dynamic_ncols=True)
    for idx, item in enumerate(pbar_apply):
        results[idx] = pool.apply_async(extract_feature,
                                        (Path(item),))

    # get results
    pbar = tqdm(results.items(), dynamic_ncols=True)
    for idx, result in pbar:
        feature = result.get()
        np.save(hparams.path_feature[kind_data] / f'feature_{idx}.npy',
                feature)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('kind_data', choices=('train', 'test'))
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1)
    args = parser.parse_args()

    kind_data = args.kind_data
    if not hparams.path_dataset[kind_data].exists():
        os.makedirs(hparams.path_dataset[kind_data])

    if not hparams.path_feature[kind_data].exists():
        os.makedirs(hparams.path_feature[kind_data])

    num_workers = args.num_workers

    main()
