import random

import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from .utils import calculate_accuracy, calculate_accuracy_by_labels, calculate_FP, calculate_FP_Max
from .trainer import Trainer
from catalyst import dl
from torch.utils.data import DataLoader, TensorDataset
from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, F1ScoreCallback, EarlyStoppingCallback
import csv


class CustomRunner(dl.Runner):



    def _handle_batch(self, batch):

        sx, targets = batch
        targets=targets.long()
        if self.is_train_loader:
            mode = 'train'
        else:
            mode = 'eval'

        loss, accuracy = self.model(sx, mode)
        loss = loss.mean()
        #if mode == 'train' or mode == 'eval':
        #    loss, CE_loss, E_loss, lstm_loss = self.add_regularization(loss)
        #    loss = loss.mean()


        self.batch_metrics.update(
            {"loss": loss,
             "accuracy": accuracy}
        )
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def add_regularization(self, loss):
        #print('in regularization')
        reg = 1e-3
        E_loss = 0.
        lstm_loss = 0.
        attn_loss = 0.
        CE_loss = loss

        for name, param in self.model.lstm.named_parameters():
            if 'bias' not in name:
                lstm_loss += (reg * torch.sum(torch.abs(param)))
        #
        for name, param in self.model.attn.named_parameters():
            if 'bias' not in name:
                attn_loss += (reg * torch.sum(torch.abs(param)))


        loss = loss  + lstm_loss + attn_loss
        return loss, CE_loss, E_loss, lstm_loss



class LSTMTrainer(Trainer):
    def __init__(self, model, lstm, config, device, wandb=None, trial="", crossv="", gtrial=""):
        super().__init__("encoder", wandb, device)
        self.config = config
        self.model=model
        self.lstm = lstm
        self.patience = self.config["patience"]
        self.dropout = nn.Dropout(0.65).to(device)
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.sample_number = config['sample_number']
        self.path = config['path']
        self.fig_path = config['fig_path']
        self.p_path = config['p_path']
        self.device = device
        self.gain = config['gain']
        self.temp = config['temperature']
        self.train_epoch_loss, self.train_batch_loss, self.eval_epoch_loss, self.eval_batch_loss, self.eval_batch_accuracy, self.train_epoch_accuracy = [], [], [], [], [], []
        self.train_epoch_roc, self.eval_epoch_roc = [], []
        self.eval_epoch_CE_loss, self.eval_epoch_E_loss, self.eval_epoch_lstm_loss = [], [], []
        self.test_accuracy = 0.
        self.test_auc = 0.
        self.test_loss = 0.
        self.trials = trial
        self.gtrial = gtrial
        self.max = 0
        self.min = 10
        self.exp = config['exp']
        self.cv = crossv


        self.optimizer = torch.optim.Adam(list(self.model.attn.parameters()) + list(self.model.lstm.parameters()) + list(
            self.model.encoder.parameters()) + list(
            self.model.classifier1.parameters()), lr=config['lr'], eps=1e-5)



    def train(self):
        # TODO: Make it work for all modes, right now only it defaults to pcl.
        callbacks = [
            EarlyStoppingCallback(patience=15, metric='loss', minimize=True, min_delta=0),
        ]

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min')
        train_dataset = TensorDataset(self.tr_eps, torch.arange(self.tr_eps.shape[0]))
        val_dataset = TensorDataset(self.val_eps, torch.arange(self.val_eps.shape[0]))
        runner = CustomRunner()
        v_bs = self.val_eps.shape[0]
        loaders = {
            "train": DataLoader(train_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True),
            "valid": DataLoader(val_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True),
        }

        model = self.model
        num_features = 2
        # model training
        train_loader_param = {"batch_size": 64,
                              "shuffle": True,
                              }
        val_loader_param = {"batch_size": 32,
                            "shuffle": True,
                            }

        loaders_params = {"train": train_loader_param,
                          "valid": val_loader_param}

        # datasets = {
        #               "batch_size": 64,
        #               "num_workers": 1,
        #               "loaders_params": loaders_params,
        #               "get_datasets_fn": self.datasets_fn,
        #               "num_features": num_features,

        #          },

        runner.train(
            model=model,
            optimizer=self.optimizer,
            scheduler=scheduler,
            loaders=loaders,
            callbacks=callbacks,
            logdir="./logs",
            num_epochs=self.epochs,
            verbose=True,
            distributed=False,
            load_best_on_end=True,
            main_metric="loss",

        )

    def pre_train(self, tr_eps, val_eps, tst_eps):
        self.tr_eps = tr_eps
        self.val_eps = tst_eps
        self.tst_eps = tst_eps
        #self.run_demo(self.train, 1)
        #utils.distributed_cmd_run(self.train)
        self.train()



