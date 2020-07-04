import random

import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import calculate_accuracy, Cutout, calculate_accuracy_by_labels, calculate_FP, calculate_FP_Max
from .trainer import Trainer
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from catalyst import dl
from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, F1ScoreCallback, EarlyStoppingCallback
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist




class CustomRunner(dl.Runner):

    def predict_batch(self, batch):
        #    # model inference step
        epoch_loss, accuracy, steps, epoch_acc, epoch_roc = 0., 0., 0, 0., 0.
        epoch_loss1, epoch_loss2, epoch_accuracy, epoch_FP = 0., 0., 0., 0.
        epoch = 1
        mode = 'test'

        sx , targets = batch.dataset.tensors
        sx = sx.to(self.device)
        targets = targets.to(self.device)

        logits = self.model(sx, mode)


        targets = targets.long()
        loss = F.cross_entropy(logits, targets)

        loss = loss.mean()

        accuracy, roc = self.acc_and_auc(logits, mode, targets)
        epoch_roc += roc

        epoch_loss += loss.detach().item()
        epoch_accuracy += accuracy.detach().item()
        self.log_results( epoch_loss, epoch_accuracy, epoch_roc, prefix=mode)

        return epoch_accuracy, epoch_roc, epoch_loss


    def _handle_batch(self, batch):

        sx, targets = batch
        targets=targets.long()
        if self.is_train_loader:
            mode = 'train'
        else:
            mode = 'eval'
        logits = self.model(sx, mode)
        loss = F.cross_entropy(logits, targets)
        loss = loss.mean()
        if mode == 'train' or mode == 'eval':
            loss, CE_loss, E_loss, lstm_loss = self.add_regularization(loss)

        loss = loss.mean()

        self.output = {"logits": logits}
        y_onehot = torch.FloatTensor(sx.shape[0], 2)
        y_onehot.zero_()
        y_onehot[np.arange(sx.shape[0]), targets] = 1
        self.input = {"features": sx, "targets": targets, "targets_one_hot": y_onehot}
        self.batch_metrics.update(
            {"loss": loss}
        )
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def acc_and_auc(self, logits, mode, targets):

        sig = torch.softmax(logits, dim=1).to(self.device)
        values, indices = sig.max(1)
        roc = 0.
        acc = 0.
        # y_scores = sig.detach().gather(1, targets.to(self.device).long().view(-1,1))
        if mode == 'eval' or mode == 'test':
            y_scores = sig.to(self.device).detach()[:, 1]
            roc = roc_auc_score(targets.to('cpu'), y_scores.to('cpu'))
        accuracy = calculate_accuracy_by_labels(indices, targets.to(self.device))

        return accuracy, roc

    def get_attention(self, outputs):
        weights_list = []
        for X in outputs:
            result = [torch.cat((X[i], X[-1]), 0) for i in range(X.shape[0])]
            result = torch.stack(result)
            result_tensor = self.model['attn'](result)
            weights_list.append(result_tensor)

        weights = torch.stack(weights_list)

        weights = weights.squeeze()

        normalized_weights = F.softmax(weights, dim=1)


        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)

        attn_applied = attn_applied.squeeze()
        logits = self.model['decoder'](attn_applied)
        #print("attention decoder ", time.time() - t)
        return logits

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

    def log_results(self, epoch_loss, epoch_test_accuracy,epoch_roc, prefix=""):
        print(
            "{}  Epoch Loss: {}, Epoch Accuracy: {}, roc: {},  {}".format(
                prefix.capitalize(),
                epoch_loss,
                epoch_test_accuracy,
                epoch_roc,
                prefix.capitalize()))

class LSTMTrainer(Trainer):
    def __init__(self, model, config, device, tr_labels, val_labels, test_labels, wandb=None, trial="", crossv="", gtrial=""):
        super().__init__("encoder", wandb, device)
        self.config = config
        self.model = model
        self.tr_labels = tr_labels
        self.test_labels = test_labels
        self.val_labels = val_labels
        self.tr_eps = ""
        self.val_eps = ""
        self.tst_eps = ""
        self.patience = self.config["patience"]
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.sample_number = config['sample_number']
        self.path = config['path']
        self.oldpath = config['oldpath']
        self.PT = config['pre_training']
        self.device = device
        self.gain = config['gain']
        self.train_epoch_loss, self.train_batch_loss, self.eval_epoch_loss, self.eval_batch_loss, self.eval_batch_accuracy, self.train_epoch_accuracy = [], [], [], [], [], []
        self.train_epoch_roc, self.eval_epoch_roc = [], []
        self.eval_epoch_CE_loss, self.eval_epoch_E_loss, self.eval_epoch_lstm_loss = [], [], []
        self.test_accuracy = 0.
        self.test_auc = 0.
        self.test_loss = 0.
        self.trials = trial
        self.gtrial = gtrial
        self.exp = config['exp']
        self.cv = crossv

        if self.exp in ['UFPT', 'NPT']:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'], eps=1e-5)
        else:
            if self.PT in ['milc-fMRI', 'variable-attention', 'two-loss-milc']:
                self.optimizer = torch.optim.Adam(list(self.model.decoder.parameters()),lr=config['lr'], eps=1e-5)
            else:
                self.optimizer = torch.optim.Adam(list(self.model.decoder.parameters()) + list(self.model.attn.parameters())
                                                       + list(self.model.lstm.parameters()), lr=config['lr'], eps=1e-5)


    def datasets_fn(self, num_features: int):

        tdataset = TensorDataset(self.tr_eps, self.tr_labels)
        vdataset = TensorDataset(self.val_eps,self.val_labels)
        return {"train": tdataset, "valid": vdataset}


    def train(self):

        callbacks = [
            EarlyStoppingCallback(patience=15, metric='loss', minimize=True, min_delta=0),
            AccuracyCallback(num_classes=2),
            AUCCallback(num_classes=2, input_key="targets_one_hot"),

        ]

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min')
        train_dataset = TensorDataset(self.tr_eps, self.tr_labels)
        val_dataset = TensorDataset(self.val_eps, self.val_labels)
        test_dataset = TensorDataset(self.tst_eps, self.test_labels)
        runner = CustomRunner()
        v_bs = self.val_eps.shape[0]
        t_bs = self.tst_eps.shape[0]
        loaders = {
            "train": DataLoader(train_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True),
            "valid": DataLoader(val_dataset, batch_size=v_bs, num_workers=1, shuffle=True),
        }

        model = self.model
        num_features=2
        # model training
        train_loader_param = {"batch_size": 64,
                              "shuffle":True,
                              }
        val_loader_param = {"batch_size": 32,
                              "shuffle": True,
                              }

        loaders_params = {"train" : train_loader_param,
                          "valid": val_loader_param}

        #datasets = {
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
            main_metric = "loss",

        )

        loader = DataLoader(test_dataset, batch_size=t_bs, num_workers=1, shuffle=True),

        self.test_accuracy, self.test_auc, self.test_loss = runner.predict_batch(next(iter(loader)))




    def pre_train(self, tr_eps, val_eps, tst_eps):
        self.tr_eps = tr_eps
        self.val_eps = val_eps
        self.tst_eps = tst_eps
        #self.run_demo(self.train, 1)
        #utils.distributed_cmd_run(self.train)
        self.train()
        return self.test_accuracy, self.test_auc, self.test_loss, 0

