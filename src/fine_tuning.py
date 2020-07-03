import random

import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from .utils import calculate_accuracy, Cutout, calculate_accuracy_by_labels, calculate_FP, calculate_FP_Max
from .trainer import Trainer
from src.utils import EarlyStopping
from torchvision import transforms
import matplotlib.pylab as plt
import matplotlib.pyplot as pl
import torchvision.transforms.functional as TF
from torch.autograd import Variable

class Classifier(nn.Module):
    def __init__(self, num_inputs1, num_inputs2):
        super().__init__()
        self.network = nn.Bilinear(num_inputs1, num_inputs2, 1)

    def forward(self, x1, x2):
        return self.network(x1, x2)


class FineTuning(Trainer):
    def __init__(self, encoder, config, device, tr_labels, test_labels, wandb=None):
        super().__init__(encoder, wandb, device)
        self.config = config
        self.tr_labels = tr_labels
        self.test_labels = test_labels
        self.patience = self.config["patience"]
        self.classifier1 = nn.Linear(self.encoder.hidden_size, 128).to(device)  # x1 = global, x2=patch, n_channels = 32
        self.classifier2 = nn.Linear(128, 128).to(device)
        self.classifier3 = nn.Linear(256, 2).to(device)
        self.classifier4 = nn.Sequential(
            nn.Linear(self.encoder.hidden_size, 200).to(device),
            nn.ReLU(),
            nn.Linear(200, 200).to(device),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(200, 2).to(device)

        )

        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.sample_number = config['sample_number']
        self.path = config['path']
        self.device = device
        self.train_epoch_loss , self.train_batch_loss, self.test_epoch_loss , self.test_batch_loss = [], [], [], []
        self.optimizer = torch.optim.Adam(list(self.classifier4.parameters()) ,lr=config['lr'], eps=1e-5)
        # self.optimizer = torch.optim.SGD(list(self.classifier4.parameters()), lr=config['lr'], momentum=0.9)
        self.early_stopper = EarlyStopping(patience=self.patience, verbose=False, wandb=self.wandb, name="encoder", path=self.path)
        self.transform = transforms.Compose([Cutout(n_holes=1, length=80)])

    def generate_batch(self, episodes, mode):
        if self.sample_number == 0:
            total_steps = sum([len(e) for e in episodes])
        else:
            total_steps = self.sample_number
        print('Total Steps: {}'.format(total_steps))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=True, num_samples=total_steps),
                               self.batch_size, drop_last=True)

        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]

            # print('indicesShape = ', indices.shape)
            #
            # print('printts_number = ', ts_number)
            ts_number = torch.LongTensor(indices)
            # print(mode)
            # print('tsShape = ', ts_number.shape)
            # print('ts_number = ', ts_number)
            # print('tr_index = ', self.test_index)
            # if mode == 'train':
            #     # real_ts_number = self.tr_index[indices]
            # elif mode == 'test':
            #     real_ts_number = self.test_index[indices]
            #     print('ts_numbertest = ', ts_number)
            #     print('indicestest = ', indices)

            x_t, x_tprev, x_that, ts, thats = [], [], [], [], []
            for episode in episodes_batch:
                # Get one sample from this episode
                t, t_hat = 0, 0
                t, t_hat = np.random.randint(0, len(episode)), np.random.randint(1, len(episode))
                x_t.append(episode[t])

                x_tprev.append(episode[t - 1])
                ts.append([t])
            yield torch.stack(x_t).to(self.device) / 255., torch.stack(x_tprev).to(self.device) / 255., ts_number.to(self.device)

    def do_one_epoch(self, epoch, episodes, mode):
        # mode = "train" if self.encoder.training and self.classifier1.training else "val"
        epoch_loss, accuracy, steps = 0., 0., 0
        accuracy1, accuracy2, accuracy, FP = 0., 0., 0., 0.
        epoch_loss1, epoch_loss2, epoch_accuracy, epoch_FP = 0., 0., 0., 0.
        data_generator = self.generate_batch(episodes, mode)
        for x_t, x_tprev, ts_number in data_generator:
            loss=0.

            if mode == 'train':
                targets = self.tr_labels[ts_number]

            else:
                targets = self.test_labels[ts_number]

            #if mode == 'test':
            #    print('rts = ', ts_number)
            #    print('target = ', targets)
            targets = targets.long()
            f_t_maps = self.encoder(x_t, fmaps=True)
            # Loss 1: Global at time t, f5 patches at time t-1
            f_t = f_t_maps['out']
            logits = self.classifier4(f_t)
            N = x_t.size(0)

            sig1 = torch.zeros(N, 2).to(self.device)
            loss = F.cross_entropy(logits, targets.to(self.device))
            reg = 1e-3
            for name, param in self.classifier4.named_parameters():
                if 'bias' not in name:
                     loss +=  (reg * torch.sum(torch.abs(param)))

            # print(loss)
            # print('loss shape', loss.shape)
            # print('l1 shape',l1_norm.shape)
            # print(type(loss))
            # print(type(l1_norm))

            if mode == "test":
                sig1 = torch.softmax(logits, dim=1)
                self.test_batch_loss.append(loss)
            else:
                self.train_batch_loss.append(loss)
                # print(targets.shape)

            if mode == "test":
                sig = sig1
                values, indices = sig.max(1)
                accuracy = calculate_accuracy_by_labels(indices, targets.to(self.device))

            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss
            if mode == "test":
                epoch_accuracy += accuracy.detach().item()
                # epoch_FP += FP.detach().item()
            # preds1 = torch.sigmoid(self.classifier1(x1, x2).squeeze())
            # accuracy1 += calculate_accuracy(preds1, target)
            # preds2 = torch.sigmoid(self.classifier2(x1_p, x2_p).squeeze())
            # accuracy2 += calculate_accuracy(preds2, target)
            steps += 1
        if mode == "test":
            self.test_epoch_loss.append(epoch_loss / steps)
        else:
            self.train_epoch_loss.append(epoch_loss / steps)
        self.log_results(epoch, epoch_loss1 / steps, epoch_loss / steps, epoch_accuracy / steps,
                         epoch_FP / steps, prefix=mode)
        # if mode == "test":
        #     self.early_stopper(-epoch_loss / steps, self.encoder)

    def train(self, tr_eps, val_eps, tst_eps):
        # TODO: Make it work for all modes, right now only it defaults to pcl.
        for e in range(self.epochs):
            self.encoder.eval(),  self.classifier4.train()
            mode = "train"
            self.do_one_epoch(e, tr_eps, mode)

            self.encoder.eval(), self.classifier4.eval()
            # mode="val"
            # self.do_one_epoch(e, val_eps, mode)
            mode = "test"
            self.do_one_epoch(e, tst_eps, mode)
            if self.early_stopper.early_stop:
                break
        # torch.save(self.encoder.state_dict(), os.path.join(self.path, self.config['env_name'] + '.pt'))
        # print('train epoch loss = ', self.train_epoch_loss)
        # print('train batch loss = ', self.train_batch_loss)
        #
        # print('test epoch loss = ', self.test_epoch_loss)
        # print('test batch loss = ', self.test_batch_loss)
        # pl.interactive(False)
        # diff = 307
        # c_diff = 0
        # n_test_batch = 81
        # test_batch_index = 0
        # size = 388 * (e+1)
        # train_xdim , test_xdim = [], []
        # for a in range(size):
        #     if test_batch_index < n_test_batch:
        #         test_batch_index += 1
        #         test_xdim.append(a)
        #     elif test_batch_index == n_test_batch:
        #         c_diff += 1
        #         if c_diff >= diff:
        #             c_diff = 0
        #             test_batch_index = 0
        #     train_xdim.append(a)



        f = pl.figure()
        pl.plot( self.train_batch_loss, label='train batch loss')
        pl.plot( self.test_batch_loss, label='test batch loss')


        pl.xlabel('Batches')
        pl.ylabel('Loss')
        pl.legend()
        pl.show()
        f.savefig("batch_loss_FBIRN.pdf", bbox_inches='tight')
        # print(os.path.join(self.path, self.config['env_name'] + '.pt'))

    def log_results(self, epoch_idx, epoch_loss1, epoch_loss, epoch_test_accuracy, epoch_FP, prefix=""):
        print("{} Epoch: {}, Epoch Loss: {}, Epoch Accuracy: {}, Epoch FP: {}, {}".format(prefix.capitalize(), epoch_idx, epoch_loss,
                                                                            epoch_test_accuracy,
                                                                            epoch_FP,
                                                                            prefix.capitalize()))


