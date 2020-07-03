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
import torchvision.transforms.functional as TF


class Classifier(nn.Module):
    def __init__(self, num_inputs1, num_inputs2):
        super().__init__()
        self.network = nn.Bilinear(num_inputs1, num_inputs2, 1)

    def forward(self, x1, x2):
        return self.network(x1, x2)


class InfoNCESpatioTemporalTrainer(Trainer):
    def __init__(self, encoder, config, device, wandb=None):
        super().__init__(encoder, wandb, device)
        self.config = config
        self.patience = self.config["patience"]
        self.classifier1 = nn.Linear(self.encoder.hidden_size, 128).to(device)  # x1 = global, x2=patch, n_channels = 32
        self.classifier2 = nn.Linear(128, 128).to(device)
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.sample_number = config['sample_number']
        self.path = config['path']
        self.device = device
        self.encoder_backup = self.encoder
        self.lstm_backup = ""
        self.tfilename = 'outputFILE'

        self.output_path = os.path.join(os.getcwd(), 'Output')
        self.output_path = os.path.join(self.output_path, self.tfilename)
        self.optimizer = torch.optim.Adam(list(self.classifier1.parameters()) + list(self.encoder.parameters()) +
                                          list(self.classifier2.parameters()),
                                          lr=config['lr'], eps=1e-5)
        self.early_stopper = EarlyStopping(self.encoder_backup, self.lstm_backup, patience=self.patience, verbose=False, wandb=self.wandb, name="encoder", path=self.path)
        self.transform = transforms.Compose([Cutout(n_holes=1, length=80)])

    def generate_batch(self, episodes):
        if self.sample_number == 0:
            total_steps = sum([len(e) for e in episodes])
        else:
            total_steps = self.sample_number
        # print('Total Steps: {}'.format(total_steps))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=True, num_samples=total_steps),
                               self.batch_size, drop_last=True)
        for indices in sampler:
            ts_number = []
            episodes_batch = [episodes[x] for x in indices]
            ts_number = torch.IntTensor(indices)
            x_t, x_tprev, x_that, ts, thats = [], [], [], [], []
            for episode in episodes_batch:
                # Get one sample from this episode
                t, t_hat = 0, 0
                t, t_hat = np.random.randint(1, len(episode)), np.random.randint(1, len(episode))
                x_t.append(episode[t])

                x_tprev.append(episode[t - 1])
                ts.append([t])
            yield torch.stack(x_t).to(self.device) / 255., torch.stack(x_tprev).to(self.device) / 255., ts_number.to(
                self.device)

    def do_one_epoch(self, epoch, episodes, mode):
        # mode = "train" if self.encoder.training and self.classifier1.training else "val"
        epoch_loss, accuracy, steps = 0., 0., 0
        accuracy1, accuracy2, accuracy, FP = 0., 0., 0., 0.
        epoch_loss1, epoch_loss2, epoch_accuracy, epoch_FP = 0., 0., 0., 0.
        data_generator = self.generate_batch(episodes)
        counter=0



        for x_t, x_tprev, ts_number in data_generator:
            counter = counter +1
            print(counter)

            f_t_maps, f_t_prev_maps = self.encoder(x_t, fmaps=True), self.encoder(x_tprev, fmaps=True)
            # Loss 1: Global at time t, f5 patches at time t-1
            f_t, f_t_prev = f_t_maps['out'], f_t_prev_maps['f5']
            sy = f_t_prev.size(1)
            # sx = f_t_prev.size(2)

            N = f_t.size(0)
            loss1 = 0.
            sig1 = torch.zeros(N, N).to(self.device)
            sig2 = torch.zeros(N, N).to(self.device)
            sig = torch.zeros(N, N).to(self.device)

            for y in range(sy):
                # for x in range(sx):
                predictions = self.classifier1(f_t)
                positive = f_t_prev[:, y, :]
                logits = torch.matmul(predictions, positive.t())
                logits = logits.to(self.device);
                step_loss = F.cross_entropy(logits, torch.arange(N).to(self.device))
                if mode == "test":
                    sig1 += torch.softmax(logits,dim=1)
                loss1 += step_loss
            # loss1 = loss1 / (sx * sy)

            loss1 = loss1 / (sy)

            # Loss 2: f5 patches at time t, with f5 patches at time t-1
            f_t = f_t_maps['f5']
            loss2 = 0.
            for y in range(sy):
                # for x in range(sx):
                predictions = self.classifier2(f_t[:, y, :])
                positive = f_t_prev[:, y, :]
                logits = torch.matmul(predictions, positive.t())
                logits = logits.to(self.device);
                step_loss = F.cross_entropy(logits, torch.arange(N).to(self.device))
                loss2 += step_loss
                if mode == "test":
                    sig2 += torch.softmax(logits,dim=1)
            # loss2 = loss2 / (sx * sy)
            loss2 = loss2 / (sy)
            loss = loss1 + loss2
            if mode == "test":
                sig1 = sig1 / sy
                sig2 = sig2 / sy
                sig = sig1 + sig2
                sig = torch.softmax(sig,dim=1)
                values, indices = sig.max(1)
                accuracy = calculate_accuracy_by_labels(indices, torch.arange(N).to(self.device))
                # FP = calculate_FP(sig, ts_number)
                #FP = calculate_FP_Max(indices, ts_number)
                # FP = calculate_FP(sig, ts_number)

            # print("values = ", values)
            # print("indices = ", indices)
            # print("labels = ", torch.arange(N))
            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.detach().item()
            epoch_loss1 += loss1.detach().item()
            epoch_loss2 += loss2.detach().item()
            if mode == "test":
                epoch_accuracy += accuracy.detach().item()
                # epoch_FP += FP.detach().item()
            # preds1 = torch.sigmoid(self.classifier1(x1, x2).squeeze())
            # accuracy1 += calculate_accuracy(preds1, target)
            # preds2 = torch.sigmoid(self.classifier2(x1_p, x2_p).squeeze())
            # accuracy2 += calculate_accuracy(preds2, target)
            steps += 1
        if epoch % 1 == 0:
            self.log_results(epoch, epoch_loss1 / steps, epoch_loss2 / steps, epoch_loss / steps, epoch_accuracy / steps,
                            epoch_FP / steps, prefix=mode)
            output_text_file = open(self.output_path, "a+")
            output_text_file.write("Epoch = %d, loss = %f \r\n" % (epoch, epoch_loss / steps))
            output_text_file.close()
        if mode == "test":
            self.early_stopper(epoch_loss / steps, 0, self.encoder, "lstm", 0)
        return epoch_loss / steps

    def train(self, tr_eps, val_eps, tst_eps):
        # TODO: Make it work for all modes, right now only it defaults to pcl.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        saved = 0
        for e in range(self.epochs):
            self.encoder.train(), self.classifier1.train(), self.classifier2.train()
            mode = "train"
            val_loss = self.do_one_epoch(e, tr_eps, mode)

            self.encoder.eval(), self.classifier1.eval(), self.classifier2.eval()
            # mode="val"
            # self.do_one_epoch(e, val_eps, mode)
            mode = "test"
            val_loss = self.do_one_epoch(e, tst_eps, mode)
            scheduler.step(val_loss)
            if self.early_stopper.early_stop:
                self.early_stopper(0, 0, self.encoder, "lstm", 1)
                saved = 1
                break

        if saved == 0:
            self.early_stopper(0, 0, self.encoder, "lstm", 1)
            saved = 1

        torch.save(self.encoder.state_dict(), os.path.join(self.path, self.config['env_name'] + '.pt'))

        print(os.path.join(self.path, self.config['env_name'] + '.pt'))

    def log_results(self, epoch_idx, epoch_loss1, epoch_loss2, epoch_loss, epoch_test_accuracy, epoch_FP, prefix=""):
        print("{} Epoch: {}, Epoch Loss: {}, Epoch Accuracy: {}, Epoch FP: {}, {}".format(prefix.capitalize(), epoch_idx, epoch_loss,
                                                                            epoch_test_accuracy,
                                                                            epoch_FP,
                                                                            prefix.capitalize()))


