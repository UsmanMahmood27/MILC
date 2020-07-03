import time
from collections import deque
from itertools import chain

import numpy as np
import torch
import sys
import os
from src.dim_baseline import DIMTrainer
from src.global_infonce_stdim import GlobalInfoNCESpatioTemporalTrainer
from src.global_local_infonce import GlobalLocalInfoNCESpatioTemporalTrainer
from src.spatio_temporal import SpatioTemporalTrainer
from src.utils import get_argparser
from src.cpc import CPCTrainer
from src.vae import VAETrainer
from src.no_action_feedforward_predictor import NaFFPredictorTrainer
from src.infonce_spatio_temporal import InfoNCESpatioTemporalTrainer
import wandb
import pandas as pd
from aari.episodes import get_episodes
from scripts.run_contrastive import train_encoder
import matplotlib.pyplot as pl



if __name__ == "__main__":

    # parser = get_argparser()
    # print('1')
    # args = parser.parse_args()
    # print('2')
    # tags = ['pretraining-only']
    # print('3')
    # wandb.init(project=args.wandb_proj, entity="curl-atari", tags=tags)
    # print('4')
    config = {}
    # print('5')
    # config.update(vars(args))
    # print('6')
    # wandb.config.update(config)
    # print('7')
    index_array = torch.randperm(857)
    # v = np.arange(0, 3030)
    index_array=index_array.numpy()
    np.savetxt("index_array_fMRI.csv", index_array, delimiter=",")
    # np.savetxt("index_array_labelled_KW2.csv", v, delimiter=",")
    print('8')
    values=2
    # filename = 'val_acc6.csv'
    # # print(filename)
    # df = pd.read_csv(filename, header=None)
    # d = df.values
    # d=d.T
    # d = d.reshape(values)
    # filename = 'val_auc6.csv'
    # # print(filename)
    # df = pd.read_csv(filename, header=None)
    # dd = df.values
    # dd=dd.T
    # dd = dd.reshape(values)
    # f = pl.figure()
    # #
    # pl.plot(d, label='acc')
    # pl.plot(dd, label='auc')
    # #
    #
    # pl.xlabel('epochs')
    # pl.ylabel('auc/acc')
    # pl.legend()
    # pl.show()
    # f.savefig('auc6.png', bbox_inches='tight')
    #
    # filename = 'val_loss6.csv'
    # # print(filename)
    # df = pd.read_csv(filename, header=None)
    # loss = df.values
    # loss = loss.T
    # loss = loss.reshape(values)
    #
    # filename = 'val_CE_loss6.csv'
    # # print(filename)
    # df = pd.read_csv(filename, header=None)
    # closs = df.values
    # closs = closs.T
    # closs = closs.reshape(values)
    #
    # filename = 'val_E_loss6.csv'
    # # print(filename)
    # df = pd.read_csv(filename, header=None)
    # eloss = df.values
    # eloss = eloss.T
    # eloss = eloss.reshape(values)

    # filename = 'train_loss9.csv'
    # # print(filename)
    # df = pd.read_csv(filename, header=None)
    # data = df.values[:,1:]
    # data = data.T
    # print(data.shape)
    # # lloss = lloss.reshape(values)
    #
    #
    # f = pl.figure()
    # #
    # pl.semilogy(data[:,2], label='val_total_loss')
    # pl.semilogy(data[:,5], label='val_CE_loss')
    # pl.semilogy(data[:,6], label='val_Enc_loss')
    # pl.semilogy(data[:,7], label='val_lstm_loss')
    # #
    #
    # pl.xlabel('epochs')
    # pl.ylabel('loss')
    # pl.legend()
    # pl.show()
    # f.savefig('loss9.png', bbox_inches='tight')
