import os
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import numpy as np
import sklearn
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import Tensor
from torchsummary import summary
from tqdm import tqdm

import data_manager
import evaluation
from hparams import hparams
from model import VAE
from utils import print_to_file


# Wrapper class to run PyTorch model


class Runner(object):
    def __init__(self, hparams, train_size: int, class_weight: Optional[Tensor] = None):
        # model, criterion
        self.model = VAE()

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=hparams.learning_rate,
                                          eps=hparams.eps,
                                          weight_decay=hparams.weight_decay
                                          )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    **hparams.scheduler)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        # self.kld = nn.KLDivLoss(reduction='sum')
        # device
        device_for_summary = self.__init_device(hparams.device, hparams.out_device)

        # summary
        self.writer = SummaryWriter(logdir=hparams.logdir)
        # TODO: fill in ~~DUMMY~~INPUT~~SIZE~~
        path_summary = Path(self.writer.logdir, 'summary.txt')
        if not path_summary.exists():
            print_to_file(path_summary,
                          summary,
                          (self.model, (40, 11)),
                          dict(device=device_for_summary)
                          )

        # save hyperparameters
        path_hparam = Path(self.writer.logdir, 'hparams.txt')
        if not path_hparam.exists():
            print_to_file(path_hparam, hparams.print_params)

    def __init_device(self, device, out_device):
        if device == 'cpu':
            self.in_device = torch.device('cpu')
            self.out_device = torch.device('cpu')
            self.str_device = 'cpu'
            return 'cpu'

        # device type: List[int]
        if type(device) == int:
            device = [device]
        elif type(device) == str:
            device = [int(device[-1])]
        else:  # sequence of devices
            if type(device[0]) != int:
                device = [int(d[-1]) for d in device]

        self.in_device = torch.device(f'cuda:{device[0]}')

        if len(device) > 1:
            if type(out_device) == int:
                self.out_device = torch.device(f'cuda:{out_device}')
            else:
                self.out_device = torch.device(out_device)
            self.str_device = ', '.join([f'cuda:{d}' for d in device])

            self.model = nn.DataParallel(self.model,
                                         device_ids=device,
                                         output_device=self.out_device)

        else:
            self.out_device = self.in_device
            self.str_device = str(self.in_device)

        self.model.cuda(self.in_device)
        self.bce.cuda(self.out_device)  ##

        torch.cuda.set_device(self.in_device)

        return 'cuda'

    # Running model for train, test and validation.
    def run(self, dataloader, mode: str, epoch: int):
        self.model.train() if mode == 'train' else self.model.eval()
        if mode == 'test':
            state_dict = torch.load(Path(self.writer.logdir, f'{epoch}.pt'), map_location='cpu')
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
            path_test_result = Path(self.writer.logdir, f'test_{epoch}')
            os.makedirs(path_test_result, exist_ok=True)
        else:
            path_test_result = None

        avg_loss = 0.
        y = []
        y_est = []
        pred_prob = []

        pbar = tqdm(dataloader, desc=f'{mode} {epoch:3d}', postfix='-', dynamic_ncols=True)

        for i_batch, batch in enumerate(pbar):
            # data
            x = batch['batch_x']
            x = x.to(self.in_device)  # B, F, T

            # forward
            reconstruct_x, mu, logvar = self.model(x)

            # loss
            BCE = self.bce(reconstruct_x, x.view(-1, 440)).mean(dim=1)  # (B,)
            if mode != 'test':
                loss = torch.mean(BCE - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            else:
                loss = 0.

            if mode == 'train':
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss = loss.item()

            elif mode == 'valid':
                loss = loss.item()

            else:
                y += batch['batch_y']
                y_est += (BCE < 0.5).int().tolist()
                pred_prob += BCE.tolist()

            pbar.set_postfix_str('')

            avg_loss += loss

        avg_loss = avg_loss / len(dataloader.dataset)

        y = np.array(y)
        y_est = np.array(y_est)
        pred_prob = np.array(pred_prob, dtype=np.float32)

        return avg_loss, (y, y_est, pred_prob)

    def step(self, valid_loss: float, epoch: int):
        """

        :param valid_loss:
        :param epoch:
        :return: test epoch or 0
        """
        # self.scheduler.step()
        self.scheduler.step(valid_loss)

        # print learning rate
        for param_group in self.optimizer.param_groups:
            self.writer.add_scalar('learning rate', param_group['lr'], epoch)

        if epoch % 5 == 0:
            torch.save(
                (self.model.module.state_dict()
                 if isinstance(self.model, nn.DataParallel)
                 else self.model.state_dict(),),
                Path(hparams.logdir) / f'VAE_{epoch}.pt'
            )
        return 0


def main(test_epoch: int):
    train_loader, valid_loader, test_loader = data_manager.get_dataloader(hparams)
    if test_epoch == -1:
        runner = Runner(hparams,
                        len(train_loader.dataset))

        dict_custom_scalars = dict(loss=['Multiline', ['loss/train', 'loss/valid']],)
        test_dict_custom_scalars = dict(spectrogram=['Multiline',
                                                     ['spectrogram/input',
                                                      'spectrogram/reconstructed']],)

        runner.writer.add_custom_scalars(dict(training=dict_custom_scalars))
        runner.writer.add_custom_scalars(dict(test=test_dict_custom_scalars))

        epoch = 0
        test_epoch_or_zero = 0
        print(f'Training on {runner.str_device}')
        for epoch in range(hparams.num_epochs):
            # training
            train_loss, _ = runner.run(train_loader, 'train', epoch)
            runner.writer.add_scalar('loss/train', train_loss, epoch)

            # validation
            valid_loss, _ = runner.run(valid_loader, 'valid', epoch)
            runner.writer.add_scalar('loss/valid', valid_loss, epoch)

            # check stopping criterion
            test_epoch_or_zero = runner.step(valid_loss, epoch)
            if test_epoch_or_zero > 0:
                break

        if isinstance(runner.model, nn.DataParallel):
            state_dict = runner.model.module.state_dict()
        else:
            state_dict = runner.model.state_dict()
        torch.save(state_dict, Path(runner.writer.logdir, f'{epoch}.pt'))

        print('Training Finished')
        test_epoch = test_epoch_or_zero if test_epoch_or_zero > 0 else epoch
    else:
        runner = Runner(hparams, len(test_loader.dataset))

    # test
    _, evaluate = runner.run(test_loader, 'test', test_epoch)
    y, y_est, pred_prob = evaluate
    pred_prob = np.stack((pred_prob, 1-pred_prob), axis=1)

    roc_auc = sklearn.metrics.roc_auc_score(y, pred_prob[:, 0])
    fig_roc = evaluation.draw_roc_curve(y, pred_prob)
    fig_confusion_mat = evaluation.draw_confusion_mat(y, y_est)

    runner.writer.add_scalar(f'roc auc', roc_auc)
    runner.writer.add_figure(f'roc curve', fig_roc)
    runner.writer.add_figure(f'confusion matrix', fig_confusion_mat)
    runner.writer.close()

    print(sklearn.metrics.classification_report(y, y_est))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test', type=int, default=-1)

    args = hparams.parse_argument(parser)
    test_epoch = args.test
    if test_epoch == -1:
        # check overwrite or not
        if list(Path(hparams.logdir).glob('events.out.tfevents.*')):
            while True:
                s = input(f'"{hparams.logdir}" already has tfevents. continue? (y/n)\n')
                if s.lower() == 'y':
                    shutil.rmtree(hparams.logdir)
                    os.makedirs(hparams.logdir)
                    break
                elif s.lower() == 'n':
                    exit()

    main(test_epoch)
