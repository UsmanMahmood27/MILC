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
from src.encoders import NatureCNN, ImpalaCNN, NatureOneCNN
from src.cpc import CPCTrainer
from src.vae import VAETrainer
from src.no_action_feedforward_predictor import NaFFPredictorTrainer
from src.infonce_spatio_temporal import InfoNCESpatioTemporalTrainer
import wandb
import pandas as pd
from aari.episodes import get_episodes
from scripts.run_contrastive import train_encoder








if __name__ == "__main__":
    parser = get_argparser()
    print('1')
    args = parser.parse_args()
    print('2')
    tags = ['pretraining-only']
    print('3')
    wandb.init(project=args.wandb_proj, entity="curl-atari", tags=tags)
    print('4')
    config = {}
    print('5')
    config.update(vars(args))
    print('6')
    wandb.config.update(config)
    print('7')
    index_array = torch.randperm(823)
    print('8')
    train_encoder(args,index_array)