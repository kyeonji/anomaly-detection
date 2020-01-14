import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Sequence, Union


@dataclass
class HParams(object):
    """
    If you don't understand 'field(init=False)' and __post_init__,
    read python 3.7 dataclass documentation
    """
    # Dataset Settings
    path_raw_data: Dict[str, Path] = field(init=False)
    path_dataset: Dict[str, Path] = field(init=False)
    path_feature: Dict[str, Path] = field(init=False)

    # Feature Parameters
    sample_rate: int = 16000
    fft_size: int = 512
    win_size: int = 512
    hop_size: int = 256
    n_mels: int = 40
    refresh_normconst: bool = False
    audio_length: int = sample_rate*6
    context_win: int = 11

    # summary path
    logdir: str = './runs/test1'

    # Model Parameters
    model: Dict[str, Any] = field(init=False)

    # Training Parameters
    scheduler: Dict[str, Any] = field(init=False)
    train_ratio: float = 0.85
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-8
    eps: float = 1e-8

    # Device-dependent Parameters
    # 'cpu', 'cuda:n', the cuda device no., or the tuple of the cuda device no.
    device: Union[int, str, Sequence[str], Sequence[int]] = 0
    out_device: Union[int, str] = 0
    num_workers: int = 0  # should be 0 in Windows

    def __post_init__(self):
        self.path_raw_data = dict(train=Path('testdataset/TrainData_Normal_16k'))
        self.path_dataset = dict(train=Path('traindataset'),
                                 test=Path('testdataset/evaluation_data'))
        self.path_feature = dict(train=Path('feature/Train'),
                                 test=Path('feature/Test'))

        self.model = dict()
        self.scheduler = dict()

    # Function for parsing argument and set hyper parameters
    def parse_argument(self, parser=None, print_argument=True):
        if not parser:
            parser = argparse.ArgumentParser()
        dict_self = asdict(self)
        for k in dict_self:
            parser.add_argument(f'--{k}', default='')

        args = parser.parse_args()
        for k in dict_self:
            parsed = getattr(args, k)
            if parsed == '':
                continue
            if type(dict_self[k]) == str:
                setattr(self, k, parsed)
            else:
                v = eval(parsed)
                if isinstance(v, type(dict_self[k])):
                    setattr(self, k, eval(parsed))

        if print_argument:
            self.print_params()

        return args

    def print_params(self):
        print('-------------------------')
        print('Hyper Parameter Settings')
        print('-------------------------')
        for k, v in asdict(self).items():
            print(f'{k}: {v}')
        print('-------------------------')


hparams = HParams()
