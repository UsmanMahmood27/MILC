"""
CNN+LSTM+Attention
Real Data (fBIRN + ABIDE)
Saliency
"""

from __future__ import print_function

import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.utils_attention_realData import EarlyStopping
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as tn
from torch.utils.data import BatchSampler, RandomSampler
from torchvision import transforms
import torchvision.transforms.functional as TF

from .trainer import Trainer
from .utils_attention_realData import (
    calculate_accuracy,
    calculate_accuracy_by_labels,
    calculate_false_positive,
    Cutout,
)

"""
Loss and accuracy for train
Loss acc and auc for test
Dont calc auc for train, only for test and val
"""

grads = {}


def save_grad(name):
    def hook(grad):
        grads[name] = grad

    return hook


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class LSTMTrainer(Trainer):
    def __init__(
        self,
        encoder,
        lstm,
        config,
        device,
        exp,
        tr_labels,
        val_labels,
        test_labels,
        wandb=None,
    ):
        super().__init__(encoder, wandb, device)
        self.config = config
        self.patience = self.config["patience"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.device = device
        self.batch_train_loss = []
        self.batch_val_loss = []
        self.path = config["path"]
        self.fig_path = config["fig_path"]
        self.exp = exp  # Experiment name

        self.lstm = lstm
        self.tr_labels = tr_labels
        self.test_labels = test_labels
        self.val_labels = val_labels

        # self.attn = nn.Linear(2*self.lstm.hidden_dim, 1)

        self.attn = nn.Sequential(
            nn.Linear(2 * self.lstm.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(device)

        self.decoder = nn.Sequential(
            nn.Linear(self.lstm.hidden_dim, 200), nn.ReLU(), nn.Linear(200, 2)
        ).to(device)
        self.init_weight()
        self.dropout = nn.Dropout(0.65).to(self.device)

        if self.exp in ["unfrozen", "full"]:
            self.optimizer = torch.optim.Adam(
                list(self.decoder.parameters())
                + list(self.attn.parameters())
                + list(self.lstm.parameters())
                + list(self.encoder.parameters()),
                lr=config["lr"],
                eps=1e-5,
            )
        else:
            self.optimizer = torch.optim.Adam(
                list(self.decoder.parameters())
                + list(self.attn.parameters())
                + list(self.lstm.parameters()),
                lr=config["lr"],
                eps=1e-5,
            )
        self.early_stopper = EarlyStopping(
            patience=self.patience,
            epochs=self.epochs,
            verbose=False,
            wandb=self.wandb,
            encname=self.encoder,
            lstmname=self.lstm,
            attnname=self.attn,
            decodename=self.decoder,
            path=self.path,
        )
        self.transform = transforms.Compose([Cutout(n_holes=1, length=80)])

    def init_weight(self):
        for name, param in self.decoder.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param, gain=0.5)
        for name, param in self.attn.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param, gain=0.5)

    def generate_batch(self, episodes, mode):
        total_steps = len(
            episodes
        )  # How many samples will be generated in total
        print("Total Steps: {}".format(total_steps))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        if mode == "test" or mode == "val":
            BS = len(episodes)
        else:
            BS = 16
        sampler = BatchSampler(
            RandomSampler(range(len(episodes)), replacement=False),
            BS,
            drop_last=False,
        )

        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]
            ts_number = torch.LongTensor(indices)
            i = 0
            subjects = []
            for episode in episodes_batch:
                # Get all samples from this episode
                subjects.append(episode)
            yield torch.stack(subjects).to(self.device), ts_number.to(
                self.device
            )

    def calculate_labels(self, indices_m, mode):
        targets = indices_m
        targets = Variable(targets)

        if mode == "train":
            targets = self.tr_labels[indices_m]

        elif mode == "val":
            targets = self.val_labels[indices_m]

        elif mode == "test":
            targets = self.test_labels[indices_m]

        targets = targets.long()

        return targets

    def add_regularizer(self, loss):
        reg = 1e-3
        # if self.exp != 'frozen':
        #     for name, param in self.encoder.named_parameters():
        #         if 'bias' not in name:
        #             loss += (reg * torch.sum(torch.abs(param)))

        for name, param in self.lstm.named_parameters():
            if "bias" not in name:
                loss += reg * torch.sum(torch.abs(param))

        return loss

    def compute_saliency_maps(self, X, y, subjects_per_group, trial_no):

        # Wrap the input tensors in Variables
        # self.lstm.train()
        X_var = Variable(X, requires_grad=True)
        y_var = Variable(y, requires_grad=False)

        # mode = "train"

        inputs = [self.encoder(x, fmaps=False) for x in X_var]

        new_input = torch.stack(inputs).to(self.device)
        outputs = self.lstm(new_input)

        weights_list = []

        for X in outputs:
            result = [
                self.attn(torch.cat((X[i], X[-1]), 0)) for i in range(121)
            ]
            result_tensor = torch.stack(result).to(self.device)
            weights_list.append(result_tensor)

        weights = torch.stack(weights_list).to(self.device)
        weights = weights.squeeze().to(self.device)

        print("Weights Shape:", weights.shape)

        # SoftMax normalization on weights
        normalized_weights = F.softmax(weights, dim=1)

        # Batch-wise multiplication of weights and lstm outputs

        attn_applied = torch.bmm(
            normalized_weights.unsqueeze(1).to("cpu"), outputs.to("cpu")
        )
        attn_applied = attn_applied.squeeze().to(self.device)

        # print('After attention shape:', attn_applied.shape)

        # Pass the weighted output to decoder
        output = self.decoder(attn_applied)

        grad_outputs = torch.zeros(len(X_var), 2).to(self.device)

        grad_outputs[:, y_var] = 1
        # output.register_hook(save_grad('output'))

        self.optimizer.zero_grad()
        start_time = time.time()

        # These three lines are newly written

        # output=output.gather(1, y_var.view(-1, 1)).squeeze()
        # ones_tensor = torch.ones(len(y_var))
        # output.backward(ones_tensor)

        output.backward(gradient=grad_outputs)
        elapsed_time = time.time() - start_time

        saliency = X_var.grad.data
        print("Saliency Shape:", saliency.shape)
        print("Backward Elapsed time:", elapsed_time)
        saliency = saliency.cpu().clone().numpy()

        path = os.path.join(self.path, "Saliency")
        filename = os.path.join(
            path,
            self.exp
            + "_subj_"
            + str(subjects_per_group)
            + "_trial_"
            + str(trial_no)
            + ".npz",
        )

        with open(filename, "wb") as file:
            np.save(file, saliency)
            print("Saliency saved successfully...")
            print("Location:", filename)
        return saliency

    def do_one_epoch(
        self, epoch, episodes, mode, subjects_per_group, trial_no
    ):
        # mode = "train" if self.encoder.training and self.classifier1.training else "val"
        epoch_loss1, epoch_loss2, epoch_accuracy, epoch_roc, steps = (
            0.0,
            0.0,
            0.0,
            0.0,
            0,
        )
        accuracy1, accuracy2, accuracy = 0.0, 0.0, 0.0
        roc = 0.0
        data_generator = self.generate_batch(episodes, mode)
        for subjects, indices_m in data_generator:
            loss = 0.0
            inputs = [self.encoder(x, fmaps=False) for x in subjects]
            new_input = torch.stack(inputs).to(self.device)
            outputs = self.lstm(new_input)

            print("Outputs From LSTM:", outputs.shape)

            weights_list = []

            for X in outputs:
                result = [
                    self.attn(torch.cat((X[i], X[-1]), 0)) for i in range(121)
                ]
                result_tensor = torch.stack(result).to(self.device)
                weights_list.append(result_tensor)

            weights = torch.stack(weights_list).to(self.device)
            weights = weights.squeeze().to(self.device)

            # print('Weights Shape:', weights.shape)

            # SoftMax normalization on weights
            normalized_weights = F.softmax(weights, dim=1)

            # Batch-wise multiplication of weights and lstm outputs

            attn_applied = torch.bmm(
                normalized_weights.unsqueeze(1).to("cpu"), outputs.to("cpu")
            )
            attn_applied = attn_applied.squeeze().to(self.device)

            # print('After attention shape:', attn_applied.shape)

            # Pass the weighted output to decoder
            logits = self.decoder(attn_applied)
            indices_m = Variable(indices_m)
            labels = self.calculate_labels(indices_m, mode)
            loss = F.cross_entropy(logits, labels.to(self.device))

            epoch_loss1 += loss.detach().item()
            # if mode == 'train' or mode == 'val':
            #    loss = self.add_regularizer(loss)
            N = logits.size(0)  # No. of samples in batch
            sig = torch.zeros(N, 2).to(self.device)
            sig = torch.softmax(logits, dim=1)
            values, indices = sig.max(1)
            accuracy = calculate_accuracy_by_labels(
                indices, labels.to(self.device)
            )

            if mode == "val":
                targets = labels
                y_scores = sig.to(self.device).detach()[:, 1]
                roc = roc_auc_score(targets.to("cpu"), y_scores.to("cpu"))

            if mode == "val":
                self.batch_val_loss.append(loss)
            elif mode == "train":
                self.batch_train_loss.append(loss)

            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss2 += loss.detach().item()
            epoch_accuracy += accuracy.detach().item()
            if mode == "val":
                epoch_roc += roc
            steps += 1

        self.log_results(
            subjects_per_group,
            trial_no,
            epoch,
            epoch_loss1 / steps,
            epoch_loss2 / steps,
            epoch_accuracy / steps,
            epoch_roc / steps,
            prefix=mode,
        )

        if mode == "val":
            self.early_stopper(
                epoch,
                -epoch_loss2 / steps,
                epoch_accuracy / steps,
                self.encoder,
                self.lstm,
                self.attn,
                self.decoder,
            )
            if epoch % 10 == 0:
                print(
                    "Best Epoch So Far: {}, Best Loss So Far: {}, Best Accuracy So Far: {}".format(
                        self.early_stopper.best_epoch,
                        -self.early_stopper.val_loss_max,
                        self.early_stopper.best_acc,
                    )
                )

        return epoch_accuracy / steps, epoch_loss1 / steps, epoch_loss2 / steps

    def save_predictions(self, indices, labels, subjects_per_group, trial_no):
        predictions = indices
        indices = indices.cpu().numpy()
        indices = pd.DataFrame(indices)

        labels = labels.cpu().numpy()
        labels = pd.DataFrame(labels)

        result = pd.concat([indices, labels], axis=1)
        path = os.path.join(self.path, "Predictions")
        filename = os.path.join(
            path,
            self.exp
            + "_subj_"
            + str(subjects_per_group)
            + "_trial_"
            + str(trial_no)
            + ".csv",
        )
        result.to_csv(filename)

    def do_test(self, epoch, episodes, mode, subjects_per_group, trial_no):

        epoch_loss, epoch_accuracy, epoch_roc, steps = 0.0, 0.0, 0.0, 0
        accuracy1, accuracy2, accuracy = 0.0, 0.0, 0.0
        data_generator = self.generate_batch(episodes, mode)
        start_time = time.time()
        for subjects, indices_m in data_generator:
            loss = 0.0
            X_var = Variable(subjects, requires_grad=True)
            inputs = [self.encoder(x, fmaps=False) for x in X_var]
            new_input = torch.stack(inputs).to(self.device)
            outputs = self.lstm(new_input)

            weights_list = []

            for X in outputs:
                result = [
                    self.attn(torch.cat((X[i], X[-1]), 0)) for i in range(121)
                ]
                result_tensor = torch.stack(result).to(self.device)
                weights_list.append(result_tensor)

            weights = torch.stack(weights_list).to(self.device)
            weights = weights.squeeze().to(self.device)

            # print('Weights Shape:', weights.shape)
            normalized_weights = F.softmax(weights, dim=1)

            attn_applied = torch.bmm(
                normalized_weights.unsqueeze(1).to("cpu"), outputs.to("cpu")
            )
            attn_applied = attn_applied.squeeze().to(self.device)
            logits = self.decoder(attn_applied)

            indices_m = Variable(indices_m)
            labels = self.calculate_labels(indices_m, mode)
            loss = F.cross_entropy(logits, labels.to(self.device))

            # self.optimizer.zero_grad()
            # print('We are at the point of gradient calc..')
            # loss.backward()
            # print('Gradient during Test:', X_var.grad)

            N = logits.size(0)  # No. of samples in batch
            sig = torch.zeros(N, 2).to(self.device)
            sig = torch.softmax(logits, dim=1)
            values, indices = sig.max(1)
            accuracy = calculate_accuracy_by_labels(
                indices, labels.to(self.device)
            )
            self.save_predictions(
                indices, labels, subjects_per_group, trial_no
            )
            targets = labels
            y_scores = sig.to(self.device).detach()[:, 1]
            roc = roc_auc_score(targets.to("cpu"), y_scores.to("cpu"))
            saliency = self.compute_saliency_maps(
                subjects, labels, subjects_per_group, trial_no
            )
            print("Saliency Shape:", saliency.shape)

            epoch_loss += loss.detach().item()
            epoch_accuracy += accuracy.detach().item()
            epoch_roc += roc
            steps += 1
        elapsed_time = time.time() - start_time
        print("Total Test Time:", elapsed_time)
        return epoch_accuracy / steps, epoch_loss / steps, epoch_roc / steps

    def train(self, tr_eps, val_eps, tst_eps, subjects_per_group, trial_no):
        # TODO: Make it work for all modes, right now only it defaults to pcl.
        test_accuracy = 0.0
        req_epoch = self.epochs - 1
        training_loss1 = torch.zeros(self.epochs)
        training_loss2 = torch.zeros(self.epochs)
        training_accuracy = torch.zeros(self.epochs)

        val_loss1 = torch.zeros(self.epochs)
        val_loss2 = torch.zeros(self.epochs)

        val_accuracy = torch.zeros(self.epochs)

        self.lstm.train(), self.attn.train(), self.decoder.train()
        if self.exp == "frozen":
            self.encoder.eval()
        else:
            self.encoder.train()

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[50, 120], gamma=0.35
        )
        for e in range(self.epochs):
            mode = "train"
            (
                training_accuracy[e],
                training_loss1[e],
                training_loss2[e],
            ) = self.do_one_epoch(
                e, tr_eps, mode, subjects_per_group, trial_no
            )
            mode = "val"
            with torch.no_grad():
                (
                    val_accuracy[e],
                    val_loss1[e],
                    val_loss2[e],
                ) = self.do_one_epoch(
                    e, val_eps, mode, subjects_per_group, trial_no
                )
            scheduler.step()
            if self.early_stopper.early_stop:
                req_epoch = e
                break

        self.save_loss_accuracy(
            training_accuracy,
            training_loss1,
            training_loss2,
            val_accuracy,
            val_loss1,
            val_loss2,
            subjects_per_group,
            trial_no,
        )
        self.load_model()
        print("Now One batch test (do test) is running.....")
        mode = "test"
        test_accuracy, test_loss, test_auc = self.do_test(
            e, tst_eps, mode, subjects_per_group, trial_no
        )
        print(
            "Test Accuracy: {}, Test Loss: {}, Test Area Under Curve: {}".format(
                test_accuracy, test_loss, test_auc
            )
        )
        # self.plot_loss_accuracy(training_accuracy, val_accuracy, req_epoch)
        return test_accuracy, test_loss, test_auc, req_epoch

    def log_results(
        self,
        subjects_per_group,
        trial_no,
        epoch_idx,
        epoch_loss1,
        epoch_loss2,
        epoch_accuracy,
        roc,
        prefix="",
    ):
        print(
            "{} Subj Per Group: {}, Trial: {}, Epoch: {}, Epoch Loss1: {}, Epoch Loss2: {}, Epoch Accuracy: {}, AUC: {}, {}".format(
                prefix.capitalize(),
                subjects_per_group,
                trial_no,
                epoch_idx,
                epoch_loss1,
                epoch_loss2,
                epoch_accuracy,
                roc,
                prefix.capitalize(),
            )
        )

    def save_loss_accuracy(
        self,
        training_accuracy,
        training_loss1,
        training_loss2,
        val_accuracy,
        val_loss1,
        val_loss2,
        subjects_per_group,
        trial_no,
    ):
        training_accuracy = training_accuracy.numpy()
        training_accuracy = pd.DataFrame(training_accuracy)
        path = os.path.join(self.path, "TrainAccuracy")
        filename = os.path.join(
            path,
            self.exp
            + "_subj_"
            + str(subjects_per_group)
            + "_trial_"
            + str(trial_no)
            + ".csv",
        )
        training_accuracy.to_csv(filename)

        training_loss1 = training_loss1.numpy()
        training_loss1 = pd.DataFrame(training_loss1)
        path = os.path.join(self.path, "TrainLoss")

        # CE means cross-entropy loss
        filename = os.path.join(
            path,
            self.exp
            + "_CE_"
            + "_subj_"
            + str(subjects_per_group)
            + "_trial_"
            + str(trial_no)
            + ".csv",
        )
        training_loss1.to_csv(filename)

        training_loss2 = training_loss2.numpy()
        training_loss2 = pd.DataFrame(training_loss2)
        path = os.path.join(self.path, "TrainLoss")

        # CEPlusL1 implies cross-entropy plus L1 regularization

        filename = os.path.join(
            path,
            self.exp
            + "_CEPlusL1_"
            + "_subj_"
            + str(subjects_per_group)
            + "_trial_"
            + str(trial_no)
            + ".csv",
        )
        training_loss2.to_csv(filename)

        val_accuracy = val_accuracy.numpy()
        val_accuracy = pd.DataFrame(val_accuracy)
        path = os.path.join(self.path, "ValAccuracy")
        filename = os.path.join(
            path,
            self.exp
            + "_subj_"
            + str(subjects_per_group)
            + "_trial_"
            + str(trial_no)
            + ".csv",
        )
        val_accuracy.to_csv(filename)

        val_loss1 = val_loss1.numpy()
        val_loss1 = pd.DataFrame(val_loss1)
        path = os.path.join(self.path, "ValLoss")
        filename = os.path.join(
            path,
            self.exp
            + "_CE_"
            + "_subj_"
            + str(subjects_per_group)
            + "_trial_"
            + str(trial_no)
            + ".csv",
        )
        val_loss1.to_csv(filename)

        val_loss2 = val_loss2.numpy()
        val_loss2 = pd.DataFrame(val_loss2)
        path = os.path.join(self.path, "ValLoss")
        filename = os.path.join(
            path,
            self.exp
            + "_CEPlusL1_"
            + "_subj_"
            + str(subjects_per_group)
            + "_trial_"
            + str(trial_no)
            + ".csv",
        )
        val_loss2.to_csv(filename)

    def load_model(self):
        encoder_dict = torch.load(
            os.path.join(self.path, "encoder.pt"), map_location=self.device
        )
        self.encoder.load_state_dict(encoder_dict)
        self.encoder.eval()
        self.encoder.to(self.device)

        lstm_dict = torch.load(
            os.path.join(self.path, "lstm.pt"), map_location=self.device
        )
        self.lstm.load_state_dict(lstm_dict)
        self.lstm.eval()
        self.lstm.to(self.device)

        attn_dict = torch.load(
            os.path.join(self.path, "attn.pt"), map_location=self.device
        )
        self.attn.load_state_dict(attn_dict)
        self.attn.eval()
        self.attn.to(self.device)

        decoder_dict = torch.load(
            os.path.join(self.path, "decoder.pt"), map_location=self.device
        )
        self.decoder.load_state_dict(decoder_dict)
        self.decoder.eval()
        self.decoder.to(self.device)

    def plot_loss_accuracy(self, training_accuracy, val_accuracy, req_epoch):
        f = plt.figure()
        plt.plot(
            self.batch_train_loss, "g", label="Training Loss", linewidth=2
        )
        plt.plot(
            self.batch_val_loss, "r", label="Validation Loss", linewidth=2
        )
        plt.xlabel("Batches")
        plt.ylabel("Loss")
        plt.title("Learning Curves")
        plt.legend()
        plt.show()
        f.savefig(os.path.join(self.fig_path, "loss.png"), bbox_inches="tight")

        f = plt.figure()
        plt.plot(
            range(req_epoch),
            training_accuracy[:req_epoch],
            "g",
            label="Training Accuracy",
            linewidth=2,
        )
        plt.plot(
            range(req_epoch),
            val_accuracy[:req_epoch],
            "r",
            label="Val Accuracy",
            linewidth=2,
        )
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Learning Curves")
        plt.legend()
        plt.show()
        f.savefig(
            os.path.join(self.fig_path, "accuracy.png"), bbox_inches="tight"
        )

    def eval_on_train(
        self, tr_eps, val_eps, test_eps, subjects_per_group, trial_no
    ):
        self.load_model()
        trial_no = 100
        print("Now One batch test (do test) is running.....")
        mode = "test"
        e = 0
        test_eps = tr_eps
        self.test_labels = self.tr_labels
        test_accuracy, test_loss, test_auc = self.do_test(
            e, test_eps, mode, subjects_per_group, trial_no
        )
        print(
            "Test Accuracy: {}, Test Loss: {}, Test Area Under Curve: {}".format(
                test_accuracy, test_loss, test_auc
            )
        )
