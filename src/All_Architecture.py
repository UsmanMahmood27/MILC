import torch
import torch.nn as nn
import torch.nn.utils.rnn as tn
from torch.autograd import Variable
import os
import torch.nn.functional as F


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

        ).to(device)


        self.init_weight()
        self.loadModels()

    def init_weight(self):
        for name, param in self.decoder.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.attn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)

    def loadModels(self):
        if self.PT in ['milc', 'variable-attention', 'two-loss-milc']:
            if self.exp in ['UFPT', 'FPT']:
                print('in ufpt and fpt')
                model_dict = torch.load(os.path.join(self.oldpath, 'lstm' + '.pt'), map_location=self.device)
                self.lstm.load_state_dict(model_dict)
                #self.model.lstm.to(self.device)

                model_dict = torch.load(os.path.join(self.oldpath, 'attn' + '.pt'), map_location=self.device)
                self.attn.load_state_dict(model_dict)
                #self.model.attn.to(self.device)

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
        logits = self.decoder(attn_applied)
        #print("attention decoder ", time.time() - t)
        return logits

    def forward(self, sx, mode='train'):

        inputs = [self.encoder(x, fmaps=False) for x in sx]
        outputs = self.lstm(inputs, mode)
        logits = self.get_attention(outputs)

        return logits
