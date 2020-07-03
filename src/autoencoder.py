'''
Autoencoder
'''

import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from a2c_ppo_acktr.utils import init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CheckSize(nn.Module):
    def forward(self, x):
        print("Size is",x.size())
        return x

class MyAutoEncoderNew(nn.Module):
    def __init__(self, input_channels, args):
        super().__init__()

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.conv1 = init_(nn.Conv1d(input_channels, 16, 3, stride=1, padding=1))
        self.conv2 = init_(nn.Conv1d(16, 8, 3, stride=1, padding=1))
        self.conv3 = init_(nn.Conv1d(8, 10, 3, stride=1, padding=1))

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.mp = nn.MaxPool1d(2, stride=1, return_indices=True)
        self.up = nn.MaxUnpool1d(2, stride=1)

        self.trans1 = init_(nn.ConvTranspose1d(10, 8, 3, stride=1, padding=1))
        self.trans2 = init_(nn.ConvTranspose1d(8, 16, 3, stride=1, padding=1))
        self.trans3 = init_(nn.ConvTranspose1d(16, input_channels, 3, stride=1, padding=1))

        self.flatten = Flatten()
        self.check = CheckSize()

        self.train()

    def forward(self, inputs, pretraining=False):
        x = self.conv1(inputs)
        x = self.relu(x)
        x, ind1 = self.mp(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        enc_out, ind2 = self.mp(x)

        code = self.flatten(enc_out)

        x = self.up(enc_out, ind2)
        x = self.relu(x)
        x = self.trans1(x)
        x = self.relu(x)
        x = self.trans2(x)
        x = self.up(x, ind1)
        x = self.relu(x)
        x = self.trans3(x)
        dec_out = self.relu(x)

        if pretraining:
            return {
                'enc': enc_out,
                'dec': dec_out
            }
        return code


class MyAutoEncoder(nn.Module):
    def __init__(self, input_channels, args):
        super().__init__()

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.conv1 = init_(nn.Conv1d(input_channels, 32, 4, stride=1))
        self.conv2 = init_(nn.Conv1d(32, 64, 4, stride=1))
        self.conv3 = init_(nn.Conv1d(64, 128, 3, stride=1))
        self.conv4 = init_(nn.Conv1d(128, 64, 2, stride=1))

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.mp = nn.MaxPool1d(2, stride=1, return_indices=True)
        self.up = nn.MaxUnpool1d(2, stride=1)

        self.trans1 = init_(nn.ConvTranspose1d(64, 128, 2, stride=1))
        self.trans2 = init_(nn.ConvTranspose1d(128, 64, 3, stride=1))
        self.trans3 = init_(nn.ConvTranspose1d(64, 32, 4, stride=1))
        self.trans4 = init_(nn.ConvTranspose1d(32, input_channels, 4, stride=1))

        self.lin1 = init_(nn.Linear(64*11, 256))
        self.lin2 = init_(nn.Linear(256, 64*11))

        self.flatten = Flatten()
        self.check = CheckSize()
        self.train()


    def forward(self, inputs, pretraining=False):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.flatten(x)
        enc_out = self.lin1(x)

        x = self.lin2(enc_out)
        x = x.view(x.size(0), 64, 11)
        x = self.trans1(x)
        x = self.relu(x)
        x = self.trans2(x)
        x = self.relu(x)
        x = self.trans3(x)
        x = self.relu(x)
        x = self.trans4(x)
        dec_out = self.relu(x)
        # self.check(x)

        if pretraining:
            return {
                'enc': enc_out,
                'dec': dec_out
            }
        return enc_out

class LinearAutoenc(nn.Module):
    def __init__(self):
        super(LinearAutoenc, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10 * 20, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 10 * 20),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


