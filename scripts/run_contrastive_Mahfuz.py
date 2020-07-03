import sys
import time
from collections import deque
from itertools import chain

import numpy as np
import torch

from src.dim_baseline import DIMTrainer
from src.global_infonce_stdim import GlobalInfoNCESpatioTemporalTrainer
from src.global_local_infonce import GlobalLocalInfoNCESpatioTemporalTrainer
from src.spatio_temporal import SpatioTemporalTrainer
from src.utils import get_argparser
from src.encoders_Mahfuz import NatureCNN, ImpalaCNN, NatureOneCNN
from src.cpc import CPCTrainer
from src.vae import VAETrainer
from src.no_action_feedforward_predictor import NaFFPredictorTrainer
from src.infonce_spatio_temporal import InfoNCESpatioTemporalTrainer
import wandb
import pandas as pd
from aari.episodes import get_episodes
from scripts.LoadSynPreTrainData import loadData
from src.encoder_slstm_attn_ import LSTMTrainer
from src.lstm_attn import subjLSTM
import datetime
import os

def train_encoder(args):
    if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        device = torch.device("cuda:" + cudaID)
        # device = torch.device("cuda:" + str(args.cuda_id))
    else:
        device = torch.device("cpu")
    print('device = ', device)
    currentDT = datetime.datetime.now()
    d1 = currentDT.strftime("%Y-%m-%d%H:%M:%S")
    dir = 'run-' + d1 + ' -sim-sub3'
    wdb = 'wandb_new'
    path = os.path.join(os.getcwd(), wdb)
    path = os.path.join(path, dir)
    args.path = path
    os.mkdir(path)

    finalData = loadData()

    print(finalData.shape)

    tr_eps = finalData[:, 0:700, :, :]
    val_eps = finalData[:, 700:900, :, :]
    test_eps = finalData[:, 900:1000, :, :]

    tr_eps = torch.from_numpy(tr_eps).float()
    val_eps = torch.from_numpy(val_eps).float()
    test_eps = torch.from_numpy(test_eps).float()
    input_data = torch.rand(20, 10, 1, 160, 210)
    # tr_eps = torch.rand(20, 10, 1, 160, 210)
    # val_eps = torch.rand(20, 10, 1, 160, 210)
    # observation_shape = tr_eps[0][0].shape

    print("trainershape", tr_eps.shape)
    print("valshape", val_eps.shape)

    initial_n_channels=1

    observation_shape = finalData.shape
    if args.encoder_type == "Nature":
        encoder = NatureCNN(observation_shape[2], args)
        # encoder = NatureCNN(input_channels, args)

    elif args.encoder_type == "Impala":
        encoder = ImpalaCNN(observation_shape[2], args)

    elif args.encoder_type == "NatureOne":
        encoder = NatureOneCNN(observation_shape[2], args)
    lstm_model = subjLSTM(device, args.feature_size, args.lstm_size, num_layers=args.lstm_layers,
                          freeze_embeddings=True, gain=0.1)
    lstm_model.to(device)
    encoder.to(device)
    torch.set_num_threads(1)

    config = {}
    config.update(vars(args))
    config['obs_space'] = observation_shape  # weird hack
    if args.method == 'cpc':
        trainer = CPCTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == 'spatial-appo':
        trainer = SpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == 'vae':
        trainer = VAETrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "naff":
        trainer = NaFFPredictorTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "infonce-stdim":
        trainer = InfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "global-infonce-stdim":
        trainer = GlobalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "sub-enc-lstm":
        trainer = LSTMTrainer(encoder, lstm_model, config, device=device, wandb=wandb)
    elif args.method == "global-local-infonce-stdim":
        trainer = GlobalLocalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "dim":
        trainer = DIMTrainer(encoder, config, device=device, wandb=wandb)
    else:
        assert False, "method {} has no trainer".format(args.method)

    # print("trainershape", tr_eps.shape)
    # print("valshape", val_eps.shape)
    trainer.train(tr_eps, val_eps, test_eps)

    return encoder


if __name__ == "__main__":
    print(sys.argv[0])
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['pretraining-only']
    #wandb.init()
    #wandb.init(project=args.wandb_proj, entity="curl-atari", tags=tags)
    config = {}
    config.update(vars(args))
    #wandb.config.update(config)
    train_encoder(args)
