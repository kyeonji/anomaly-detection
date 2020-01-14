import numpy as np
from tqdm import tqdm

from hparams import hparams

train_path_list = list(hparams.path_feature['train'].glob('*.npy'))
test_path_list = list(hparams.path_feature['test'].glob('*.npy'))

train_max = []
test_max = []
train_len = []
test_len = []

pbar = tqdm(train_path_list, dynamic_ncols=True)
for path in pbar:
    train_data = np.load(path)
    train_max.append(train_data.max())
    train_len.append(train_data.shape[1])

pbar = tqdm(test_path_list, dynamic_ncols=True)
for path in pbar:
    test_data = np.load(path)
    test_max.append(test_data.max())
    test_len.append(test_data.shape[1])

train_max = max(train_max)
test_max = max(test_max)

print(f'train_max: {train_max}\n'
      f'test_max: {test_max}')
print(train_len)
print(test_len)
