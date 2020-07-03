import torch
import torch.nn as nn
import torch.nn.functional as F
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CheckSize(nn.Module):
    def forward(self, x):
        print("final size is :   ",x.size())
        return x

class ChangeDevice(nn.Module):
    def forward(self, x, device):
        x = x.to(device)
        return x


class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            Conv2dSame(in_channels, out_channels, 3)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = F.relu(out)
        return out


class ImpalaCNN(nn.Module):
    def __init__(self, input_channels, args):
        super(ImpalaCNN, self).__init__()
        self.hidden_size = args.feature_size
        self.depths = [16, 32, 32, 32]
        self.downsample = not args.no_downsample
        self.layer1 = self._make_layer(input_channels, self.depths[0])
        self.layer2 = self._make_layer(self.depths[0], self.depths[1])
        self.layer3 = self._make_layer(self.depths[1], self.depths[2])
        self.layer4 = self._make_layer(self.depths[2], self.depths[3])
        if self.downsample:
            self.final_conv_size = 32 * 9 * 9
        else:
            self.final_conv_size = 32 * 12 * 9
        self.final_linear = nn.Linear(self.final_conv_size, self.hidden_size)
        self.flatten = Flatten()
        self.train()

    def _make_layer(self, in_channels, depth):
        return nn.Sequential(
            Conv2dSame(in_channels, depth, 3),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(),
            ResidualBlock(depth, depth),
            nn.ReLU(),
            ResidualBlock(depth, depth)
        )

    def forward(self, inputs):
        out = inputs
        if self.downsample:
            out = self.layer3(self.layer2(self.layer1(out)))
        else:
            out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        return F.relu(self.final_linear(self.flatten(out)))


class NatureCNN(nn.Module):
    def __init__(self, input_channels, args, device, device_one, device_two, device_three):
        super().__init__()
        self.feature_size = args.fMRI_feature_size
        self.device = device
        self.device_one = device_one
        self.device_two = device_two
        self.device_three = device_three
        self.hidden_size = self.feature_size
        self.deep = args.deep
        self.downsample = not args.no_downsample
        self.input_channels = input_channels
        self.two_d = args.fMRI_twoD
        self.end_with_relu = args.end_with_relu
        self.args = args
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.flatten = Flatten()

        if self.deep:
            self.final_conv_size = 32 * 11 * 13
            self.final_conv_shape = (32, 11, 13)

            self.main = nn.Sequential(
                init_(nn.Conv2d(self.input_channels, 16, (9, 10), stride=1)).to(self.device_one),
                nn.ReLU(),
                init_(nn.Conv2d(16, 16, (9, 10), stride=1)).to(self.device_one),
                nn.ReLU(),
                init_(nn.Conv2d(16, 16, (8, 9), stride=1)).to(self.device_one),
                nn.ReLU(),
                init_(nn.Conv2d(16, 16, (7, 8), stride=1)).to(self.device_one),
                nn.ReLU(),
                init_(nn.Conv2d(16, 32, (5, 6), stride=1)).to(self.device_two),
                nn.ReLU(),
                init_(nn.Conv2d(32, 32, (4, 5), stride=1)).to(self.device_two),
                nn.ReLU(),
                init_(nn.Conv2d(32, 32, (4, 5), stride=1)).to(self.device_three),
                nn.ReLU(),
                init_(nn.Conv2d(32, 32, (4, 5), stride=1)).to(self.device_three),
                nn.ReLU(),
                # init_(nn.ConvTranspose2d(32, 3, (10,11), stride=1)).to(self.device),
                # nn.ReLU(),
                # CheckSize(),
                Flatten(),
                init_(nn.Linear(self.final_conv_size, self.feature_size)).to(self.device),
                # nn.ReLU()
            )

        elif self.two_d:
            self.final_conv_size = 16 * 24 * 30
            self.final_conv_shape = (16, 24, 30)
            # self.main = nn.Sequential(
            #     init_(nn.Conv2d(self.input_channels, 32, (9,10), stride=1)),
            #     nn.ReLU(),
            #     init_(nn.Conv2d(32, 64, (9,10), stride=1)),
            #     nn.ReLU(),
            #     init_(nn.Conv2d(64, 128, (8,9), stride=1)),
            #     nn.ReLU(),
            #     init_(nn.Conv2d(128, 128, (7,8), stride=1)),
            #     nn.ReLU(),
            #     Flatten(),
            #     init_(nn.Linear(self.final_conv_size, self.feature_size))
            #     #nn.ReLU()
            # )
            self.main = nn.Sequential(
                init_(nn.Conv2d(self.input_channels, 16, (9, 10), stride=1)).to(self.device_one),
                nn.ReLU(),
                init_(nn.Conv2d(16, 16, (9, 10), stride=1)).to(self.device_two),
                nn.ReLU(),
                init_(nn.Conv2d(16, 16, (8, 9), stride=1)).to(self.device_three),
                nn.ReLU(),
                init_(nn.Conv2d(16, 16, (7, 8), stride=1)).to(self.device_three),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(self.final_conv_size, self.feature_size)).to(self.device),
                # nn.ReLU()
            )
        else:
            self.final_conv_size = 10 * 24 * 30 * 6
            self.final_conv_shape = (10, 24, 30, 6)
            #self.final_conv_size = 10 * 18 * 23 * 5
            #self.final_conv_shape = (10, 18, 23, 5)
            self.main = nn.Sequential(
                init_(nn.Conv3d(self.input_channels, 32, (9, 10, 2), stride=(1, 1, 1))),
                nn.ReLU(),
                init_(nn.Conv3d(32, 64, (9, 10, 2), stride=(1, 1, 1))),
                nn.ReLU(),
                init_(nn.Conv3d(64, 128, (8, 9, 2), stride=(1, 1, 1))),
                nn.ReLU(),
                init_(nn.Conv3d(128, 10, (7, 8, 2), stride=(1, 1, 1))),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(self.final_conv_size, self.feature_size)),



                #nn.ReLU()
            )
        self.train()

    def forward(self, inputs, fmaps=False):
        f2 = self.main[:8](inputs)
        f2 = f2.to(self.device_two)
        f6 = self.main[8:12](f2)
        f6 = f6.to(self.device_three)
        f8 = self.main[12:16](f6)
        f8 = f8.to(self.device)
        out = self.main[16:](f8)
        if self.end_with_relu:
            assert self.args.method != "vae", "can't end with relu and use vae!"
            out = F.relu(out)
        # if fmaps:
        #     return {
        #         'f5': f5.permute(0, 2, 3, 1),
        #         'f7': f7.permute(0, 2, 3, 1),
        #         'out': out
        #     }
        return out


class NatureOneCNN(nn.Module):
    def __init__(self, input_channels, args):
        super().__init__()
        self.feature_size = args.feature_size
        self.hidden_size = self.feature_size
        self.downsample = not args.no_downsample
        self.input_channels = input_channels
        self.fully_connected = False
        self.end_with_relu = args.end_with_relu
        self.args = args
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.flatten = Flatten()

        if self.downsample:
            self.final_conv_size = 32 * 7 * 7
            self.final_conv_shape = (32, 7, 7)
            self.main = nn.Sequential(
                init_(nn.Conv2d(input_channels, 32, 8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 32, 3, stride=1)),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(self.final_conv_size, self.feature_size)),
                #nn.ReLU()
            )
        elif self.fully_connected:
            self.final_conv_size = 200 * 12
            self.final_conv_shape = (200, 12)
            self.main = nn.Sequential(
                Flatten(),
                init_(nn.Linear(1060, 1064)),
                nn.ReLU(),
                init_(nn.Linear(1064, 512)),
                nn.ReLU(),
                init_(nn.Linear(512, 256)),
                nn.ReLU(),
                init_(nn.Linear(256, self.feature_size)),
                init_(nn.Linear(200, 128)),
                nn.ReLU(),
                # nn.ReLU()
            )

        else:
            self.final_conv_size = 200 * 12
            self.final_conv_shape = (200, 12)
            self.main = nn.Sequential(
                init_(nn.Conv1d(input_channels, 64, 4, stride=1)),
                nn.ReLU(),
                init_(nn.Conv1d(64, 128, 4, stride=1)),
                nn.ReLU(),
                init_(nn.Conv1d(128, 200, 3, stride=1)),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(self.final_conv_size, self.feature_size)),
                init_(nn.Conv1d(200, 128, 3, stride=1)),
                nn.ReLU(),
                # nn.ReLU()
            )
        # self.train()

    def forward(self, inputs, fmaps=False, five=False):
        f5 = self.main[:6](inputs)
        # f7 = self.main[6:8](f5)
        out = self.main[6:8](f5)
        # f5 = self.main[8:](f5)

        if self.end_with_relu:
            assert self.args.method != "vae", "can't end with relu and use vae!"
            out = F.relu(out)
        if five:
            return f5.permute(0, 2, 1)
        if fmaps:
            return {
                'f5': f5.permute(0, 2, 1),
                # 'f7': f7.permute(0, 2, 1),
                'out': out
            }
        return out
