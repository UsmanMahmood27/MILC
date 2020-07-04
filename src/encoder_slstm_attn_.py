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
from torchvision import transforms
import csv


class Classifier(nn.Module):
    def __init__(self, num_inputs1, num_inputs2):
        super().__init__()
        self.network = nn.Bilinear(num_inputs1, num_inputs2, 1)

    def forward(self, x1, x2):
        return self.network(x1, x2)


class LSTMTrainer(Trainer):
    def __init__(self, encoder, lstm, config, device, wandb=None, trial="", crossv="", gtrial=""):
        super().__init__(encoder, wandb, device)
        self.config = config
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
        self.classifier1 = nn.Sequential(
            nn.Linear(self.encoder.hidden_size, config['lstm_size']),

        ).to(device)

        self.classifier4 = nn.Sequential(
            nn.Linear(config['lstm_size'], config['lstm_size']),

        ).to(device)

        self.classifier3 = nn.Sequential(
            nn.Linear(200, 200),

        ).to(device)

        self.classifier2 = nn.Sequential(
            nn.Linear(config['lstm_size'], config['lstm_size']),
            nn.ReLU(),
            nn.Linear(config['lstm_size'], config['lstm_size'])

        ).to(device)

        self.attn = nn.Sequential(
            nn.Linear(2 * self.lstm.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)


        self.init_weight()
        self.dropout = nn.Dropout(0.65).to(self.device)

        self.optimizer = torch.optim.Adam(list(self.attn.parameters()) + list(self.lstm.parameters()) + list(
            self.encoder.parameters()) + list(
            self.classifier1.parameters()), lr=config['lr'], eps=1e-5)

        self.encoder_backup = self.encoder
        self.lstm_backup = self.lstm
        self.early_stopper = EarlyStopping(self.encoder_backup, self.lstm_backup, patience=self.patience, verbose=False,
                                           wandb=self.wandb, name="encoder",
                                           path=self.path, trial=self.trials)
        self.transform = transforms.Compose([Cutout(n_holes=1, length=80)])

    def init_weight(self):
        for name, param in self.classifier1.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.attn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)

    def generate_batch(self, episodes, mode):
        if self.sample_number == 0:
            total_steps = sum([len(e) for e in episodes])
        else:
            total_steps = self.sample_number
        # print('Total Steps: {}'.format(total_steps))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        if mode == 'train':
            BS = 32
        else:
            BS = 32

        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=False),
                               self.batch_size, drop_last=False)

        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]
            ts_number = torch.LongTensor(indices)
            i = 0
            sx = []
            sx_next = []
            for episode in episodes_batch:
                # Get all samples from this episode
                mean = episode.mean()
                sd = episode.std()
                episode = (episode - mean) / sd
                sx.append(episode)

            # tensor_sx = torch.stack(sx)
            # sx_max = tensor_sx.max()
            # sx_min = tensor_sx.min()
            # if(sx_max > self.max):
            #     self.max = sx_max
            # if(sx_min < self.min):
            #     self.min = sx_min

            yield torch.stack(sx).to(self.device) , ts_number.to(self.device)

    def do_one_epoch(self, epoch, episodes, mode):
        # mode = "train" if self.encoder.training and self.classifier1.training else "val"
        epoch_loss, accuracy, steps, epoch_acc, epoch_roc = 0., 0., 0, 0., 0.
        epoch_CE_loss, epoch_E_loss, epoch_lstm_loss = 0., 0., 0.,
        accuracy1, accuracy2, accuracy, FP = 0., 0., 0., 0.
        epoch_loss1, epoch_loss2, epoch_accuracy, epoch_FP = 0., 0., 0., 0.

        data_generator = self.generate_batch(episodes, mode)
        for sx, ts_number in data_generator:
            # print("in sx")
            # sxcopy = copy.deepcopy(sx)
            # for i in range (sx.size(0)):
            #     mean = sxcopy[i,:,:,:].mean()
            #     sd = sxcopy[i,:,:,:].std()
            #     sxcopy[i,:,:,:] = (sxcopy [i,:,:,:] - mean) / sd
            # a=1
            # zsx = stats.zscore(sx, axis=0)
            # zsx = torch.from_numpy(zsx).float()
            loss = 0.
            accuracy = 0.
            loss2 = 0.
            accuracy2 = 0.
            roc = 0.

            CE_loss, E_loss, lstm_loss = 0., 0., 0.

            # print('sx', sx.device, sx.dtype, type(sx), sx.type())
            inputs = [self.encoder(x, fmaps=False) for x in sx]

            #x_t_five = [self.encoder(x, five=True) for x in sx]

            outputs = self.lstm(inputs, mode)
            inputs_tensor = torch.stack(inputs)


            #windows_tensor = torch.stack((x_t_five))



            N = inputs_tensor.size(0)
            targets = torch.arange(N).to(self.device)
            #loss2, accuracy2 = self.window_loss(windows_tensor, mode, targets)
            # sig = torch.zeros(N, N).to(self.device)
            logits = self.get_attention(outputs)
            logits = logits.to(self.device)

            sx = inputs_tensor.size(0)
            sy = inputs_tensor.size(1)
            v = np.arange(0, sx)
            random_matrix = torch.randperm(sy)
            for loop in range(sx - 1):
                random_matrix = np.concatenate((random_matrix, torch.randperm(sy)), axis=0)
            random_matrix = np.reshape(random_matrix, (sx, sy))
            #loss3, accuracy3 = self.after_lstm_loss(outputs, logits, sx, sy, v, random_matrix, mode, targets)
            #loss4, accuracy4 = self.window_attention_loss(windows_tensor, logits, mode, targets, sy, v, random_matrix)
            for y in range(sy):
                # print("y= ", y)
                # for x in range(sx):
                y_index = random_matrix[:, y]
                positive = inputs_tensor[v, y_index, :].clone()
                positive = self.classifier1(positive)
                #logits = self.classifier2(logits)
                # positive_norms = LA.norm(positive.detach().cpu().numpy(), ord=1, axis=1).reshape(sx, 1)
                # logits_norms = LA.norm(logits.detach().cpu().numpy(), ord=1, axis=1).reshape(1, sx)
                # norms = torch.from_numpy(positive_norms * logits_norms)
                mlogits = torch.matmul(positive, logits.t())
                # mlogits = mlogits / ((self.temp * norms).to(self.device))
                mlogits = mlogits.to(self.device)
                step_loss = F.cross_entropy(mlogits, targets)
                sig = torch.softmax(mlogits, dim=1).to(self.device)
                step_acc, step_roc = self.acc_and_auc(sig, mode, targets)
                loss += step_loss
                accuracy += step_acc
            loss = loss / (sy)
            accuracy = accuracy / (sy)
            # regularization
            # if mode == 'train' or mode == 'eval':
            #    loss, CE_loss, E_loss, lstm_loss = self.add_regularization(loss)

            #print('loss = ', loss)
            #print('loss2 = ', loss2)
            #print('acc = ', accuracy)
            #print('acc2 = ', accuracy2)
            #loss = loss + loss4
            #accuracy = (accuracy + accuracy4) / 2
            #sig = sig + sig2
            # accuracy, roc = self.acc_and_auc(sig, mode, targets)
            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.detach().item()
            epoch_accuracy += accuracy.detach().item()

            # if mode == 'train' or mode == 'eval':
            #     epoch_CE_loss += CE_loss.detach().item()
            #     epoch_E_loss += E_loss
            #     epoch_lstm_loss += lstm_loss.detach().item()
            if mode == 'eval' or mode == 'test':
                epoch_roc += roc

            steps += 1

        if mode == "eval":
            self.eval_batch_accuracy.append(epoch_accuracy / steps)
            self.eval_epoch_loss.append(epoch_loss / steps)
            self.eval_epoch_roc.append(epoch_roc / steps)
            self.eval_epoch_CE_loss.append(epoch_CE_loss / steps)
            self.eval_epoch_E_loss.append(epoch_E_loss / steps)
            self.eval_epoch_lstm_loss.append(epoch_lstm_loss / steps)
        elif mode == 'train':
            self.train_epoch_loss.append(epoch_loss / steps)
            self.train_epoch_accuracy.append(epoch_accuracy / steps)
        if epoch % 1 == 0:
            self.log_results(epoch, epoch_loss1 / steps, epoch_loss / steps, epoch_accuracy / steps,
                             epoch_FP / steps, epoch_roc / steps, prefix=mode)
        if mode == "eval":
            self.early_stopper(epoch_loss / steps, epoch_accuracy / steps, self.encoder, self.lstm, self.attn,
                               self.classifier1, 0)
        if mode == 'test':
            self.test_accuracy = epoch_accuracy / steps
            self.test_auc = epoch_roc / steps
            self.test_loss = epoch_loss / steps

        return epoch_loss / steps

    def after_lstm_loss(self, outputs, logits, sx, sy, v, random_matrix, mode, targets):
        loss = 0.
        accuracy = 0.
        roc = 0.
        for y in range(sy):
            y_index = random_matrix[:, y]
            positive = outputs[v, y_index, :].clone()
            positive = self.classifier4(positive)
            mlogits = torch.matmul(positive, logits.t())
            mlogits = mlogits.to(self.device)
            step_loss = F.cross_entropy(mlogits, targets)
            sig = torch.softmax(mlogits, dim=1).to(self.device)
            step_acc, step_roc = self.acc_and_auc(sig, mode, targets)
            loss += step_loss
            accuracy += step_acc
        loss = loss / (sy)
        accuracy = accuracy / (sy)
        return loss, accuracy

    def window_attention_loss(self, WindowsTensor, logits, mode, targets, sy, v, random_matrix):
        accuracy2 =0.
        roc = 0.
        #total_t_windows = WindowsTensor.shape[0] * (WindowsTensor.shape[1] - 1)
        #random_values = torch.randperm(total_t_windows)
        #random_values_index = 0
        loss2 = 0.
        N = WindowsTensor.shape[0]
        sig2 = torch.zeros(N, N).to(self.device)
        for i in range(WindowsTensor.shape[1] - 1):
            y_index = random_matrix[:, i]
            positive = WindowsTensor[v, y_index, :, :].clone()

            sy = positive.shape[1]
            for y in range(sy):
                # for x in range(sx):
                predictions = self.classifier3 (positive[:, y, :])
                mlogits = torch.matmul(predictions, logits.t())
                mlogits = mlogits.to(self.device)
                step_loss = F.cross_entropy(mlogits, torch.arange(N).to(self.device))
                loss2 += step_loss
                # if mode == "test":
                sig2 = torch.sigmoid(mlogits)
                step_acc, step_roc = self.acc_and_auc(sig2, mode, targets)
                accuracy2 += step_acc
            # loss2 = loss2 / (sx * sy)
            loss2 = loss2 / (sy)
            accuracy2 = accuracy2 / (sy)
        loss2 = loss2 / (WindowsTensor.shape[1] - 1)
        accuracy2 = accuracy2 / (WindowsTensor.shape[1] - 1)

        return loss2, accuracy2

    def window_loss(self, WindowsTensor, mode, targets):
        accuracy2 =0.
        roc = 0.
        total_t_windows = WindowsTensor.shape[0] * (WindowsTensor.shape[1] - 1)
        random_values = torch.randperm(total_t_windows)
        random_values_index = 0
        loss2 = 0.
        N = WindowsTensor.shape[0]
        sig2 = torch.zeros(N, N).to(self.device)
        for i in range(WindowsTensor.shape[1] - 1):
            f_t = []
            f_t_next = []
            for j in range(N):
                a = random_values[random_values_index] / (WindowsTensor.shape[1] - 1)
                b = random_values[random_values_index] % (WindowsTensor.shape[1] - 1)
                f_t.append(WindowsTensor[a, b, :, :])
                f_t_next.append(WindowsTensor[a, b + 1, :, :])
                random_values_index += 1
            f_t = torch.stack(f_t)
            f_t_next = torch.stack(f_t_next)

            sy = f_t.shape[1]
            for y in range(sy):
                # for x in range(sx):
                predictions = self.classifier3 (f_t[:, y, :])
                positive = f_t_next[:, y, :]
                logits = torch.matmul(predictions, positive.t())
                logits = logits.to(self.device)
                step_loss = F.cross_entropy(logits, torch.arange(N).to(self.device))
                loss2 += step_loss
                # if mode == "test":
                sig2 = torch.sigmoid(logits)
                step_acc, step_roc = self.acc_and_auc(sig2, mode, targets)
                accuracy2 += step_acc
            # loss2 = loss2 / (sx * sy)
            loss2 = loss2 / (sy)
            accuracy2 = accuracy2 / (sy)
        loss2 = loss2 / (WindowsTensor.shape[1] - 1)
        accuracy2 = accuracy2 / (WindowsTensor.shape[1] - 1)

        return loss2, accuracy2

    def acc_and_auc(self, sig, mode, targets):
        # N = logits.size(0)
        # sig = torch.zeros(N, 2).to(self.device)
        # sig = torch.softmax(sig, dim=1).to(self.device)
        values, indices = sig.max(1)
        roc = 0.
        acc = 0.
        # y_scores = sig.detach().gather(1, targets.to(self.device).long().view(-1,1))
        # if mode == 'eval' or mode == 'test':
        #    y_scores = sig.to(self.device).detach()[:, 1]
        #    roc = roc_auc_score(targets.to('cpu'), y_scores.to('cpu'))
        # print(indices)
        # print(targets)
        accuracy = calculate_accuracy_by_labels(indices, targets.to(self.device))

        return accuracy, roc

    def get_attention(self, outputs):
        # print('Outputs From LSTM:', outputs.shape)

        weights_list = []


        for X in outputs:
            result = [torch.cat((X[i], X[-1]), 0) for i in range(X.shape[0])]
            result = torch.stack(result)
            result_tensor = self.attn(result)
            weights_list.append(result_tensor)


        weights = torch.stack(weights_list).to(self.device)
        weights = weights.squeeze().to(self.device)

        # print('Weights Shape:', weights.shape)

        # SoftMax normalization on weights
        normalized_weights = F.softmax(weights, dim=1)

        # Batch-wise multiplication of weights and lstm outputs

        attn_applied = torch.bmm(normalized_weights.unsqueeze(1).to(self.device), outputs.to(self.device))
        attn_applied = attn_applied.squeeze().to(self.device)

        # print('After attention shape:', attn_applied.shape)

        # Pass the weighted output to decoder
        # logits = self.decoder(attn_applied)
        return attn_applied

    def add_regularization(self, loss):
        reg = 1e-3
        E_loss = 0.
        lstm_loss = 0.
        CE_loss = loss
        # for name, param in self.encoder.named_parameters():
        # if 'bias' not in name:
        # E_loss += (reg * torch.sum(torch.abs(param)))

        for name, param in self.lstm.named_parameters():
            if 'bias' not in name:
                lstm_loss += (reg * torch.sum(torch.abs(param)))

        for name, param in self.encoder.named_parameters():
            if 'bias' not in name:
                lstm_loss += (reg * torch.sum(torch.abs(param)))

        for name, param in self.attn.named_parameters():
            if 'bias' not in name:
                lstm_loss += (reg * torch.sum(torch.abs(param)))

        loss = loss + E_loss + lstm_loss
        return loss, CE_loss, E_loss, lstm_loss

    def validate(self, val_eps):

        model_dict = torch.load(os.path.join(self.p_path, 'encoder' + self.trials + '.pt'), map_location=self.device)
        self.encoder.load_state_dict(model_dict)
        self.encoder.eval()
        self.encoder.to(self.device)

        model_dict = torch.load(os.path.join(self.p_path, 'lstm' + self.trials + '.pt'), map_location=self.device)
        self.lstm.load_state_dict(model_dict)
        self.lstm.eval()
        self.lstm.to(self.device)

        # model_dict = torch.load(os.path.join(self.p_path, 'decoder' + self.trials + '.pt'), map_location=self.device)
        # self.decoder.load_state_dict(model_dict)
        # self.decoder.eval()
        # self.decoder.to(self.device)

        mode = 'eval'
        self.do_one_epoch(0, val_eps, mode)
        return self.test_auc

    def load_model_and_test(self, tst_eps):
        model_dict = torch.load(os.path.join(self.path, 'encoder' + self.trials + '.pt'))
        self.encoder.load_state_dict(model_dict)
        self.encoder.eval()
        self.encoder.to(self.device)

        model_dict = torch.load(os.path.join(self.path, 'lstm' + self.trials + '.pt'))
        self.lstm.load_state_dict(model_dict)
        self.lstm.eval()
        self.lstm.to(self.device)

        mode = 'test'
        self.do_one_epoch(0, tst_eps, mode)

    def save_loss_and_auc(self):

        with open(os.path.join(self.path, 'all_data_information' + self.trials + '.csv'), 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            self.train_epoch_loss.insert(0, 'train_epoch_loss')
            wr.writerow(self.train_epoch_loss)

            self.train_epoch_accuracy.insert(0, 'train_epoch_accuracy')
            wr.writerow(self.train_epoch_accuracy)

            self.eval_epoch_loss.insert(0, 'eval_epoch_loss')
            wr.writerow(self.eval_epoch_loss)

            self.eval_batch_accuracy.insert(0, 'eval_batch_accuracy')
            wr.writerow(self.eval_batch_accuracy)

            self.eval_epoch_roc.insert(0, 'eval_epoch_roc')
            wr.writerow(self.eval_epoch_roc)

            self.eval_epoch_CE_loss.insert(0, 'eval_epoch_CE_loss')
            wr.writerow(self.eval_epoch_CE_loss)

            self.eval_epoch_E_loss.insert(0, 'eval_epoch_E_loss')
            wr.writerow(self.eval_epoch_E_loss)

            self.eval_epoch_lstm_loss.insert(0, 'eval_epoch_lstm_loss')
            wr.writerow(self.eval_epoch_lstm_loss)

    def train(self, tr_eps, val_eps, tst_eps):
        # TODO: Make it work for all modes, right now only it defaults to pcl.
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[4, 30, 128, 256, 512, 700, 800, 2500], gamma=0.15)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        saved = 0
        for e in range(self.epochs):
            self.encoder.train(), self.lstm.train()

            mode = "train"
            val_loss = self.do_one_epoch(e, tr_eps, mode)

            self.encoder.eval(), self.lstm.eval()
            mode = "eval"
            val_loss = self.do_one_epoch(e, tst_eps, mode)
            scheduler.step(val_loss)
            if self.early_stopper.early_stop:
                self.early_stopper(0, 0, self.encoder, self.lstm, self.attn, self.classifier1, 1)
                saved = 1
                break

        if saved == 0:
            self.early_stopper(0, 0, self.encoder, self.lstm, self.attn, self.classifier1, 1)
            saved = 1

        # self.save_loss_and_auc()
        # self.load_model_and_test(tst_eps)

        # f = pl.figure()
        #
        # pl.plot(self.train_epoch_loss[1:], label='train_total_loss')
        # pl.plot(self.eval_epoch_loss[1:], label='val_total_loss')
        # pl.plot(self.eval_epoch_CE_loss[1:], label='val_CE_loss')
        # pl.plot(self.eval_epoch_E_loss[1:], label='val_Enc_loss')
        # pl.plot(self.eval_epoch_lstm_loss[1:], label='val_lstm_loss')
        # # #
        # #
        # pl.xlabel('epochs')
        # pl.ylabel('loss')
        # pl.legend()
        # pl.show()
        # f.savefig(os.path.join(self.fig_path, 'all_loss.png'), bbox_inches='tight')
        #
        # f = pl.figure()
        # #
        # pl.plot(self.train_epoch_accuracy[1:], label='train_acc')
        # pl.plot(self.eval_batch_accuracy[1:], label='val_acc')
        # pl.plot(self.eval_epoch_roc[1:], label='val_auc')
        #
        # #
        #
        # pl.xlabel('epochs')
        # pl.ylabel('acc/auc')
        # pl.legend()
        # pl.show()
        # f.savefig(os.path.join(self.fig_path, 'acc.png'), bbox_inches='tight')

        print(self.early_stopper.val_acc_max)

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
