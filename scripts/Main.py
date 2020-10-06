from collections import deque
from itertools import chain
import os
import sys
import time

from aari.episodes import get_episodes
import numpy as np
import pandas as pd
from scripts.run_contrastive import train_encoder
from src.cpc import CPCTrainer
from src.dim_baseline import DIMTrainer
from src.encoders import ImpalaCNN, NatureCNN, NatureOneCNN
from src.global_infonce_stdim import GlobalInfoNCESpatioTemporalTrainer
from src.global_local_infonce import GlobalLocalInfoNCESpatioTemporalTrainer
from src.infonce_spatio_temporal import InfoNCESpatioTemporalTrainer
from src.no_action_feedforward_predictor import NaFFPredictorTrainer
from src.spatio_temporal import SpatioTemporalTrainer
from src.utils import get_argparser
from src.vae import VAETrainer
import torch
import wandb

if __name__ == "__main__":
    parser = get_argparser()
    print("1")
    args = parser.parse_args()
    print("2")
    tags = ["pretraining-only"]
    print("3")
    wandb.init(project=args.wandb_proj, entity="curl-atari", tags=tags)
    print("4")
    config = {}
    print("5")
    config.update(vars(args))
    print("6")
    wandb.config.update(config)
    print("7")
    index_array = torch.randperm(823)
    print("8")
    train_encoder(args, index_array)
