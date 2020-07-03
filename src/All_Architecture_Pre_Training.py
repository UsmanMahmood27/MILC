import torch
import torch.nn as nn
import torch.nn.utils.rnn as tn
from torch.autograd import Variable
import os
import torch.nn.functional as F
import numpy as np
from .utils import calculate_accuracy_by_labels

class combinedModel(nn.Module):
    """Bidirectional LSTM for classifying subjects."""

    def __init__(self, encoder, lstm, gain=0.1, PT="", exp="UFPT", device="cuda", oldpath=""):

        super().__init__()
        self.encoder = encoder
        self.lstm = lstm
        self.gain = gain
        self.PT = PT
        self.exp = exp
        self.device = device
        self.oldpath=oldpath
        self.attn = nn.Sequential(
            nn.Linear(2 * self.lstm.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.lstm.hidden_dim, self.lstm.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.lstm.hidden_dim, 2)

        )

        self.classifier1 = nn.Sequential(
            nn.Linear(self.encoder.feature_size, self.lstm.hidden_dim),

        )


        self.init_weight()

    def init_weight(self):
        for name, param in self.attn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)

        for name, param in self.classifier1.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)


    def get_attention(self, outputs):
        #print('in attention')
        weights_list = []
        for X in outputs:
            #t=time.time()
            result = [torch.cat((X[i], X[-1]), 0) for i in range(X.shape[0])]
            result = torch.stack(result)
            result_tensor = self.attn(result)
            weights_list.append(result_tensor)

        weights = torch.stack(weights_list)

        weights = weights.squeeze()

        normalized_weights = F.softmax(weights, dim=1)


        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)

        attn_applied = attn_applied.squeeze()
        #print("attention decoder ", time.time() - t)
        return attn_applied

    def acc_and_auc(self, sig, mode, targets):

        values, indices = sig.max(1)
        roc = 0.
        accuracy = calculate_accuracy_by_labels(indices, targets)

        return accuracy, roc
    def forward(self, sx, mode='train'):
        loss = 0.
        accuracy = 0.
        #print('sx shape', sx.shape)
        inputs = [self.encoder(x, fmaps=False) for x in sx]
        outputs = self.lstm(inputs, mode)
        logits = self.get_attention(outputs)
        inputs_tensor = torch.stack(inputs)
        N = inputs_tensor.size(0)
        targets = torch.arange(N).to(self.device)
        ssx = inputs_tensor.size(0)
        sy = inputs_tensor.size(1)
        v = np.arange(0, ssx)
        random_matrix = torch.randperm(sy)
        for loop in range(ssx - 1):
            random_matrix = np.concatenate((random_matrix, torch.randperm(sy)), axis=0)
        random_matrix = np.reshape(random_matrix, (ssx, sy))
        # loss3, accuracy3 = self.after_lstm_loss(outputs, logits, sx, sy, v, random_matrix, mode, targets)
        # loss4, accuracy4 = self.window_attention_loss(windows_tensor, logits, mode, targets, sy, v, random_matrix)
        # print('lstm device Y start  ' + torch.cuda.get_device_name(device=self.device) + ' ', (torch.cuda.memory_allocated(device=self.device) / (1024*1024)))
        # print('encoder device Y start  ' + torch.cuda.get_device_name(device=self.device_encoder) + ' ', (torch.cuda.memory_allocated(device=self.device_encoder) / (1024*1024)))
        for y in range(sy):
            # print("y= ", y)
            # for x in range(sx):
            y_index = random_matrix[:, y]
            positive = inputs_tensor[v, y_index, :].clone()
            positive = self.classifier1(positive)
            # logits = self.classifier2(logits)
            # positive_norms = LA.norm(positive.detach().cpu().numpy(), ord=1, axis=1).reshape(sx, 1)
            # logits_norms = LA.norm(logits.detach().cpu().numpy(), ord=1, axis=1).reshape(1, sx)
            # norms = torch.from_numpy(positive_norms * logits_norms)
            mlogits = torch.matmul(positive, logits.t())
            # mlogits = mlogits / ((self.temp * norms).to(self.device))
            mlogits = mlogits.to(self.device)
            step_loss = F.cross_entropy(mlogits, targets)
            sig = torch.softmax(mlogits.detach(), dim=1)
            step_acc, step_roc = self.acc_and_auc(sig, mode, targets)
            loss += step_loss
            accuracy += step_acc
            # del step_loss
            # del step_acc
            # del positive
            # del mlogits
        # print('lstm device Y end  ' + torch.cuda.get_device_name(device=self.device) + ' ', (torch.cuda.memory_allocated(device=self.device) / (1024*1024)))
        # print('encoder device Y end  ' + torch.cuda.get_device_name(device=self.device_encoder) + ' ', (torch.cuda.memory_allocated(device=self.device_encoder) / (1024*1024)))
        loss = loss / (sy)
        accuracy = accuracy / (sy)
        #print('loss', loss)
        return loss, accuracy
