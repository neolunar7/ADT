import dataset
import math
import numpy as np
import time
import torch
import torch.nn as nn
import pandas as pd
import os

from config import args
from torch.utils import data
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer, device, debug_mode, test_per_epoch, num_epochs, weight_path, train_data, test_data):
        self._device = device
        self._debug_mode = debug_mode
        self._weight_path = weight_path
        self._num_epochs = num_epochs
        self._test_per_epoch = test_per_epoch

        self._train_data = train_data
        self._test_data = test_data
        self._test_generator = data.DataLoader(dataset=test_data, shuffle=False,
                                               batch_size=args.test_batch, num_workers=args.num_workers)
        self._inference_generator = data.DataLoader(dataset=test_data, shuffle=False,
                                               batch_size=args.inference_batch, num_workers=args.num_workers)

        self._optimizer = optimizer
        self._model = model
        self._loss = nn.MSELoss(reduce=False)

        self._max_epoch = 0
        self._train_loss = 0.5
        self._test_loss = 0.5

    def _update(self, loss):
        self._optimizer.zero_grad()
        loss.mean().backward()
        self._optimizer.step()

    def _write_wandb(self):
        import wandb
        wandb.log({
            'Train Loss': self._train_loss,
            'Test Loss': self._test_loss
        })

    def _forward(self, generator, training_mode):
        mse_loss_list = []
        inference_list = []
        for src_instrument, _, src_melspectrogram in tqdm(generator):
            output_catted = self._model(src_melspectrogram.to(self._device))
            gt = src_instrument.to(self._device)            

            if training_mode != 'infer':
                mean_loss = self._loss(output_catted, gt).mean()
            elif training_mode == 'infer':
                # inference_result = output_catted.view(-1,np.shape(output)[0]).cpu().detach().numpy()
                # inference_list.append(inference_result)
                pass
                
            if training_mode == 'train':
                self._update(mean_loss)
            if training_mode != 'infer':
                mse_loss_list.append(mean_loss.item())

        if training_mode != 'infer':
            mean_mse_loss = np.mean(mse_loss_list)

            print_val = f'loss: {mean_mse_loss:.4f}'

            if training_mode == 'train':
                self._train_loss = mean_mse_loss
                print(f'[Train]     {print_val}')
            elif training_mode == 'test':
                self._test_loss = mean_mse_loss
                print(f'[Test]      {print_val}')
        if training_mode == 'infer':
            return inference_list

    def train(self):
        for epoch in range(self._num_epochs):
            print(f'\nEpoch: {epoch:03d} re-shuffling...')

            # train
            train_generator = data.DataLoader(dataset=self._train_data, shuffle=False, batch_size=args.train_batch, num_workers=args.num_workers)
            self._model.train()
            self._forward(train_generator, training_mode='train')

            # save_parameters
            cur_weight = self._model.state_dict()
            torch.save(cur_weight, f'{self._weight_path}{epoch}.pt')

            # test
            if (epoch % self._test_per_epoch) == (self._test_per_epoch-1):
                with torch.no_grad():
                    self._model.eval()
                    self._forward(self._test_generator, training_mode='test')

            # write_wandb
            if not self._debug_mode:
                self._write_wandb()
    def infer(self, save=False):
        with torch.no_grad():
            self._model.eval()
            inference_result = self._forward(self._inference_generator, training_mode='infer')
            final_inference_npy = np.ndarray((0,np.shape(inference_result[0])[1]))
            for infer_result in inference_result:
                final_inference_npy = np.concatenate((final_inference_npy,infer_result), axis=0)
            if save:
                np.save(f'./inference_result/{args.name}_pt{args.pt}.npy', final_inference_npy)