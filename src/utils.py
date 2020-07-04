import argparse
import copy
import os
import subprocess

import torch
import numpy as np
from sklearn.metrics import f1_score as compute_f1_score
from collections import defaultdict

# methods that need encoder trained before
train_encoder_methods = ['cpc', 'spatial-appo', 'vae', "naff", "infonce-stdim", "global-infonce-stdim",
                         "global-local-infonce-stdim", "dim", "sub-enc-lstm", "sub-lstm"]
probe_only_methods = ["supervised", "random-cnn", "majority", "pretrained-rl-agent"]
pre_train_encoder_methods = ['basic', 'milc', 'two-loss-milc', "variable-attention", "AE"]


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre-training', type=str,
                        default='milc',
                        choices=pre_train_encoder_methods,
                        help='Pre-Training Method to Use (default: Basic )')
    parser.add_argument("--fMRI-twoD", action='store_true', default=False)
    parser.add_argument("--deep", action='store_true', default=False)
    parser.add_argument('--path', type=str,
                        default='/data/mialab/users/umahmood1/STDIMs/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/scripts/wandb',
                        help='Path to store the encoder (default: )')
    parser.add_argument('--oldpath', type=str,
                        default='/data/mialab/users/umahmood1/STDIMs/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/scripts/wandb',
                        help='Path to store the encoder (default: )')
    parser.add_argument('--fig-path', type=str,
                        default='/data/mialab/users/umahmood1/STDIMs/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/scripts/wandb',
                        help='Path to store the encoder (default: )')
    parser.add_argument('--p-path', type=str,
                        default='/data/mialab/users/umahmood1/STDIMs/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/scripts/wandb',
                        help='Path to store the encoder (default: )')
    parser.add_argument('--exp', type=str,
                        default='UFPT',
                        help='the exp to run (default:FPT )')
    parser.add_argument('--gain', type=float,
                        default=0.1,
                        help='gain value for init (default:0.5 )')
    parser.add_argument('--temperature', type=float, default=0.25,
                        help='Temperature for division of norms in pre training')
    parser.add_argument('--script-ID', type=int, default=1,
                        help='Task Array ID')
    parser.add_argument('--teststart-ID', type=int, default=1,
                        help='Task Set Start Index ID')
    parser.add_argument('--job-ID', type=int, default=1,
                        help='Job Array ID')
    parser.add_argument('--sample-number', type=int, default=0,
                        help='Job Array ID')
    parser.add_argument('--env-name', default='MontezumaRevengeNoFrameskip-v4',
                        help='environment to train on (default: MontezumaRevengeNoFrameskip-v4)')
    parser.add_argument('--num-frame-stack', type=int, default=1,
                        help='Number of frames to stack for a state')
    parser.add_argument('--no-downsample', action='store_true', default=True,
                        help='Whether to use a linear classifier')
    parser.add_argument('--pretraining-steps', type=int, default=100000,
                        help='Number of steps to pretrain representations (default: 100000)')
    parser.add_argument('--probe-steps', type=int, default=50000,
                        help='Number of steps to train probes (default: 30000 )')
    #     parser.add_argument('--probe-test-steps', type=int, default=15000,
    #                         help='Number of steps to train probes (default: 15000 )')
    parser.add_argument('--num-processes', type=int, default=8,
                        help='Number of parallel environments to collect samples from (default: 8)')
    parser.add_argument('--method', type=str, default='sub-enc-lstm',
                        choices=train_encoder_methods + probe_only_methods,
                        help='Method to use for training representations (default: infonce-stdim)')
    parser.add_argument('--linear', action='store_true', default=True,
                        help='Whether to use a linear classifier')
    parser.add_argument('--use_multiple_predictors', action='store_true', default=False,
                        help='Whether to use multiple linear classifiers in the contrastive loss')

    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning Rate foe learning representations (default: 5e-4)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Mini-Batch Size (default: 64)')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='Number of epochs for  (default: 100)')
    parser.add_argument('--cuda-id', type=int, default=1,
                        help='CUDA device index')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed to use')
    parser.add_argument('--encoder-type', type=str, default="NatureOne", choices=["Impala", "Nature", "NatureOne"],
                        help='Encoder type (Impala or Nature or NatureOne)')
    parser.add_argument('--feature-size', type=int, default=256,
                        help='Size of features')
    parser.add_argument('--fMRI-feature-size', type=int, default=1024,
                        help='Size of features')
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--entropy-threshold", type=float, default=0.6)
    parser.add_argument("--color", action='store_true', default=False)
    parser.add_argument("--end-with-relu", action='store_true', default=False)
    parser.add_argument("--wandb-proj", type=str, default="curl-atari-neurips-scratch")
    parser.add_argument("--num_rew_evals", type=int, default=10)
    # rl-probe specific arguments
    parser.add_argument("--checkpoint-index", type=int, default=-1)

    # naff-specific arguments
    parser.add_argument("--naff_fc_size", type=int, default=2048,
                        help="fully connected layer width for naff")
    parser.add_argument("--pred_offset", type=int, default=1,
                        help="how many steps in future to predict")
    # CPC-specific arguments
    parser.add_argument('--sequence_length', type=int, default=100,
                        help='Sequence length.')
    parser.add_argument('--steps_start', type=int, default=0,
                        help='Number of immediate future steps to ignore.')
    parser.add_argument('--steps_end', type=int, default=99,
                        help='Number of future steps to predict.')
    parser.add_argument('--steps_step', type=int, default=4,
                        help='Skip every these many frames.')
    parser.add_argument('--gru_size', type=int, default=256,
                        help='Hidden size of the GRU layers.')
    parser.add_argument('--lstm_size', type=int, default=200,
                        help='Hidden size of the LSTM layers.')
    parser.add_argument('--fMRI-lstm_size', type=int, default=512,
                        help='Hidden size of the LSTM layers.')
    parser.add_argument('--gru_layers', type=int, default=2,
                        help='Number of GRU layers.')
    parser.add_argument('--lstm_layers', type=int, default=1,
                        help='Number of LSTM layers.')
    parser.add_argument("--collect-mode", type=str, choices=["random_agent", "pretrained_ppo"],
                        default="random_agent")

    parser.add_argument("--beta", default=1.0)
    # probe arguments
    parser.add_argument("--weights-path", type=str, default="None")
    parser.add_argument("--train-encoder", action='store_true', default=True)
    parser.add_argument('--probe-lr', type=float, default=3e-6)
    parser.add_argument("--probe-collect-mode", type=str, choices=["random_agent", "pretrained_ppo"],
                        default="random_agent")
    parser.add_argument('--num-runs', type=int, default=1)
    return parser


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def calculate_accuracy(preds, y):
    preds = preds >= 0.5
    labels = y >= 0.5
    acc = preds.eq(labels).sum().float() / labels.numel()
    return acc


def calculate_accuracy_by_labels(preds, y):
    acc = torch.sum(torch.eq(preds, y).float()) / len(y)
    return acc


def calculate_FP_Max(indices, ts_number):
    FP = 0.
    N = len(ts_number)
    for i in range(len(indices)):
        x = indices[i]
        if (ts_number[x] != ts_number[i]):
            FP += 1

    return FP / N


def calculate_FP(metrics, ts_number):
    FP = 0.
    FP2 = 0.
    N = 0
    unique = torch.unique(ts_number)
    for b in range(len(unique)):
        x = torch.sum(ts_number == unique[b].item()).item()
        y = torch.sum(ts_number != unique[b].item()).item()
        ml = x * y
        N += ml
    index = (metrics > 0.5).nonzero()
    cols = torch.unique(index[:, 0])
    unique_cols = torch.unique(cols)

    for i in range(len(unique_cols)):
        col_index = (index[:, 0] == unique_cols[i]).nonzero()
        col_val = index[col_index, 1]
        xt_tsindex = ts_number[unique_cols[i]].item()
        xrem_tsindex = ts_number[col_val]
        FP += torch.sum((xrem_tsindex != xt_tsindex)).item()
    # for a in range(len(index)):
    #     if ts_number[index[a][0].item()] != ts_number[index[a][1].item()]:
    #         FP = FP + 1
    # print('N = ', N)
    # print('FP = ', FP)
    # print('FP2 = ', FP2)
    FP = float(FP)
    FP = FP / N
    return torch.tensor(FP)


def calculate_multiclass_f1_score(preds, labels):
    preds = torch.argmax(preds, dim=1).detach().numpy()
    labels = labels.numpy()
    f1score = compute_f1_score(labels, preds, average="weighted")
    return f1score


def calculate_multiclass_accuracy(preds, labels):
    preds = torch.argmax(preds, dim=1)
    acc = float(torch.sum(torch.eq(labels, preds)).data) / labels.size(0)
    return acc








