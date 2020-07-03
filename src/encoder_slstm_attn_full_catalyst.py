import random

import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from .utils import calculate_accuracy, Cutout, calculate_accuracy_by_labels, calculate_FP, calculate_FP_Max
from .trainer import Trainer
from src.utils import EarlyStopping
from torchvision import transforms
import matplotlib.pylab as plt
import matplotlib.pyplot as pl
import torchvision.transforms.functional as TF
import torch.nn.utils.rnn as tn
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from scipy import stats
from sklearn.metrics import roc_auc_score
import ipdb
import csv
from numpy import linalg as LA
from torch.utils.data import DataLoader, TensorDataset
from catalyst import dl
from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, F1ScoreCallback, EarlyStoppingCallback
class Classifier(nn.Module):
    def __init__(self, num_inputs1, num_inputs2):
        super().__init__()
        self.network = nn.Bilinear(num_inputs1, num_inputs2, 1)

    def forward(self, x1, x2):
        return self.network(x1, x2)
class CustomRunner(dl.Runner):


    def _handle_batch(self, batch):
        # model train/valid step

        x, ydummy = batch


        if self.state.is_train_loader:
            mode = 'train'
        else:
            mode = 'eval'




        # loss2, accuracy2 = self.window_loss(windows_tensor, mode, targets)
        # sig = torch.zeros(N, N).to(self.device)
        loss, accuracy = self.model(x, mode)
        # logits = logits.to(self.device)
        loss = loss.mean()
        accuracy = accuracy.mean()

        self.state.batch_metrics.update(
            {"loss": loss, "accuracy01": accuracy}
        )

        if self.state.is_train_loader:
            loss.backward()
            self.state.optimizer.step()
            self.state.optimizer.zero_grad()

    def acc_and_auc(self, sig, mode, targets):

        values, indices = sig.max(1)
        roc = 0.
        accuracy = calculate_accuracy_by_labels(indices, targets)

        return accuracy, roc

    def add_regularization(self, loss):
        reg = 1e-3
        E_loss = 0.
        lstm_loss = 0.
        CE_loss = loss
        # for name, param in self.encoder.named_parameters():
        # if 'bias' not in name:
        # E_loss += (reg * torch.sum(torch.abs(param)))

        for name, param in self.module.lstm.named_parameters():
            if 'bias' not in name:
                lstm_loss += (reg * torch.sum(torch.abs(param)))

        for name, param in self.module.encoder.named_parameters():
            if 'bias' not in name:
                lstm_loss += (reg * torch.sum(torch.abs(param)))

        for name, param in self.module.attn.named_parameters():
            if 'bias' not in name:
                lstm_loss += (reg * torch.sum(torch.abs(param)))

        loss = loss + E_loss + lstm_loss
        return loss, CE_loss, E_loss, lstm_loss

    def log_results(self, epoch_idx, epoch_loss1, epoch_loss, epoch_test_accuracy, epoch_FP, epoch_roc, prefix=""):
        print(
            "{} CV: {}, Trial: {}, Epoch: {}, Epoch Loss: {}, Epoch Accuracy: {}, Epoch FP: {} roc: {},  {}".format(
                prefix.capitalize(),
                self.cv,
                self.trials,
                epoch_idx,
                epoch_loss,
                epoch_test_accuracy,
                epoch_FP,
                epoch_roc,
                prefix.capitalize()))


class LSTMTrainer(Trainer):
    def __init__(self, model, config, device, wandb=None, trial="", crossv="", gtrial=""):
        super().__init__("encoder", wandb, device)
        self.config = config
        self.model = model
        self.device_one = "device_one"
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




        self.dropout = nn.Dropout(0.65).to(self.device)

        self.optimizer = torch.optim.Adam(list(self.model.attn.parameters()) + list(self.model.lstm.parameters()) + list(
            self.model.encoder.parameters()) + list(
            self.model.classifier1.parameters()), lr=config['lr'], eps=1e-5)

        self.transform = transforms.Compose([Cutout(n_holes=1, length=80)])



    def train(self, tr_eps, tst_eps):
        # TODO: Make it work for all modes, right now only it defaults to pcl.
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[4, 30, 128, 256, 512, 700, 800, 2500], gamma=0.15)
        callbacks = [
            EarlyStoppingCallback(patience=15, metric='loss', minimize=True, min_delta=0),

        ]

        trlabels = torch.arange(tr_eps.shape[0])
        vallabels = torch.arange(tst_eps.shape[0])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min')
        train_dataset = TensorDataset(tr_eps, trlabels)
        val_dataset = TensorDataset(tst_eps, vallabels)
        runner = CustomRunner()
        #v_bs = self.val_eps.shape[0]
        #t_bs = self.tst_eps.shape[0]
        loaders = {
           "train": DataLoader(train_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True),
           "valid": DataLoader(val_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True),
        }

        model = self.model
        num_features = 2
        # model training


        runner.train(
            model=model,
            optimizer=self.optimizer,
            scheduler=scheduler,
            loaders = loaders,
            callbacks=callbacks,
            logdir="./logs",
            num_epochs=self.epochs,
            verbose=True,
            distributed=False,
            load_best_on_end=True,

        )


