"""
data_manager.py

A file that loads saved features and convert them into PyTorch DataLoader.
"""
import multiprocessing as mp
from copy import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Union

import numpy as np
import torch
from numpy import ndarray
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import DataPerDevice


class Normalization:
    """
    Calculating and saving mean/std of all mel spectrogram with respect to time axis,
    applying normalization to the spectrogram
    This is need only when you don't load all the data on the RAM
    """

    @staticmethod
    def _sum(a: ndarray) -> ndarray:
        return a.sum(axis=-1, keepdims=True)

    @staticmethod
    def _sq_dev(a: ndarray, mean_a: ndarray) -> ndarray:
        return ((a - mean_a)**2).sum(axis=-1, keepdims=True)

    @staticmethod
    def _load_data(fname: Union[str, Path], queue: mp.Queue) -> None:
        x = np.load(fname)
        queue.put(x)

    @staticmethod
    def _calc_per_data(data,
                       list_func: Sequence[Callable],
                       args: Sequence = None,
                       ) -> Dict[Callable, Any]:
        """ gather return values of functions in `list_func`

        :param list_func:
        :param args:
        :return:
        """

        if args:
            result = {f: f(data, arg) for f, arg in zip(list_func, args)}
        else:
            result = {f: f(data) for f in list_func}
        return result

    def __init__(self, mean, std):
        self.mean = DataPerDevice(mean)
        self.std = DataPerDevice(std)

    @classmethod
    def calc_const(cls, all_files: List[Path]):

        # Calculate summation & size (parallel)
        list_fn = (np.size, cls._sum)
        pool_loader = mp.Pool(2)
        pool_calc = mp.Pool(min(mp.cpu_count() - 2, 6))
        with mp.Manager() as manager:
            queue_data = manager.Queue()
            pool_loader.starmap_async(cls._load_data,
                                      [(f, queue_data) for f in all_files])
            result: List[mp.pool.AsyncResult] = []
            for _ in tqdm(range(len(all_files)), desc='mean', dynamic_ncols=True):
                data = queue_data.get()
                result.append(pool_calc.apply_async(
                    cls._calc_per_data,
                    (data, list_fn)
                ))

        result: List[Dict] = [item.get() for item in result]
        print()

        sum_size = np.sum([item[np.size] for item in result])
        sum_ = np.sum([item[cls._sum] for item in result], axis=0)
        mean = sum_ / (sum_size // sum_.size)

        print('Calculated Size/Mean')

        # Calculate squared deviation (parallel)
        with mp.Manager() as manager:
            queue_data = manager.Queue()
            pool_loader.starmap_async(cls._load_data,
                                      [(f, queue_data) for f in all_files])
            result: List[mp.pool.AsyncResult] = []
            for _ in tqdm(range(len(all_files)), desc='std', dynamic_ncols=True):
                data = queue_data.get()
                result.append(pool_calc.apply_async(
                    cls._calc_per_data,
                    (data, (cls._sq_dev,), (mean,))
                ))

        pool_loader.close()
        pool_calc.close()
        result: List[Dict] = [item.get() for item in result]
        print()

        sum_sq_dev = np.sum([item[cls._sq_dev] for item in result], axis=0)

        std = np.sqrt(sum_sq_dev / (sum_size // sum_sq_dev.size) + 1e-5)
        print('Calculated Std')

        return cls(mean, std)

    def save(self, fname: Path):
        np.savez(fname, mean=self.mean.data[ndarray], std=self.std.data[ndarray])

    # normalize and denormalize functions can accept a ndarray or a tensor.
    def normalize(self, a):
        return (a - self.mean.get_like(a)) / self.std.get_like(a)

    def normalize_(self, a):  # in-place version
        a -= self.mean.get_like(a)
        a /= self.std.get_like(a)

        return a

    def denormalize(self, a):
        return a * self.std.get_like(a) + self.mean.get_like(a)

    def denormalize_(self, a):  # in-place version
        a *= self.std.get_like(a)
        a += self.mean.get_like(a)

        return a


class CustomDataset(Dataset):
    def __init__(self, kind_data: str, hparams, normalization=None):
        self._PATH: Path = hparams.path_feature[kind_data]

        self.all_files = list(self._PATH.glob('*.npy'))
        self.all_files.sort()

        self.context_win = hparams.context_win

        T_list = []
        for path in self.all_files:
            data = np.load(path)
            tframe = data.shape[1]
            T_list.append(tframe)

        idx_to_file_seg = []
        for file_id, T in enumerate(T_list):
            idx_to_file_seg += [(file_id, i) for i in range(T-self.context_win+1)]

        self.idx_to_file_seg = idx_to_file_seg
        self.T_list = T_list

    def __getitem__(self, idx: int) -> dict:
        file_idx, seg_idx = self.idx_to_file_seg[idx]

        data = np.load(self.all_files[file_idx], mmap_mode='r')
        x = data[:, seg_idx:seg_idx+self.context_win]
        x = torch.from_numpy(x)

        if 188 - self.context_win <= seg_idx <= self.T_list[file_idx]-188:
            y = 1
        else:
            y = 0

        return dict(x=x, y=y)

    def __len__(self):
        return len(self.idx_to_file_seg)

    @staticmethod
    def custom_collate(batch: List[dict]) -> dict:

        x_list = [item['x'].permute(1, 0) for item in batch]
        y_list = [item['y'] for item in batch]
        batch_x = pad_sequence(x_list, batch_first=True)  # B, T, F
        batch_x = batch_x.permute(0, 2, 1)  # B, F, T

        return dict(batch_x=batch_x, batch_y=y_list)

    @classmethod
    def split(cls, dataset, ratio: Sequence[float]) -> Sequence:
        """ Split the dataset into `len(ratio)` datasets.

        The sum of elements of ratio must be 1,
        and only one element can have the value of -1 which means that
        it's automaticall set to the value so that the sum of the elements is 1

        :type dataset: SALAMIDataset
        :type ratio: Sequence[float]

        :rtype: Sequence[Dataset]
        """
        if not isinstance(dataset, cls):
            raise TypeError
        n_split = len(ratio)
        ratio = np.array(ratio)
        mask = (ratio == -1)
        ratio[np.where(mask)] = 0

        assert (mask.sum() == 1 and ratio.sum() < 1
                or mask.sum() == 0 and ratio.sum() == 1)
        if mask.sum() == 1:
            ratio[np.where(mask)] = 1 - ratio.sum()

        idx_data = np.cumsum(np.insert(ratio, 0, 0) * len(dataset.all_files),
                             dtype=int)
        result = [copy(dataset) for _ in range(n_split)]

        for ii in range(n_split):
            result[ii].all_files = dataset.all_files[idx_data[ii]:idx_data[ii + 1]]
            result[ii].idx_to_file_seg = dataset.idx_to_file_seg[idx_data[ii]:idx_data[ii + 1]]
            result[ii].T_list = dataset.T_list[idx_data[ii]:idx_data[ii + 1]]
        return result


# Function to load numpy data and normalize, it returns dataloader for train, valid, test
def get_dataloader(hparams, only_test=False):
    loader_kwargs = dict(batch_size=hparams.batch_size,
                         drop_last=False,
                         num_workers=hparams.num_workers,
                         pin_memory=True,
                         collate_fn=CustomDataset.custom_collate,
                         )

    test_loader_kwargs = dict(batch_size=hparams.batch_size,
                              drop_last=False,
                              num_workers=hparams.num_workers,
                              pin_memory=True,
                              collate_fn=CustomDataset.custom_collate,  # if needed
                              )

    if only_test:
        dcase = CustomDataset('train', hparams)  # load normalization consts
        train_loader = None
        valid_loader = None
    else:
        # create train / valid loaders
        dcase = CustomDataset('train', hparams)
        train_set, valid_set = CustomDataset.split(dcase, (hparams.train_ratio, -1))
        train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
        valid_loader = DataLoader(valid_set, shuffle=False, **loader_kwargs)

    # test loader
    test_set = CustomDataset('test', hparams,)
    test_loader = DataLoader(test_set, shuffle=False, **test_loader_kwargs)

    return train_loader, valid_loader, test_loader
