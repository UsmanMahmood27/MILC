import time
from collections import deque
from itertools import chain

import numpy as np
import torch
import pandas as pd

from codes.dim_baseline import DIMTrainer
from codes.global_infonce_stdim import GlobalInfoNCESpatioTemporalTrainer
from codes.global_local_infonce import GlobalLocalInfoNCESpatioTemporalTrainer
from codes.spatio_temporal import SpatioTemporalTrainer
from codes.src_utils import get_argparser
from codes.encoders import NatureCNN, ImpalaCNN
from codes.no_action_feedforward_predictor import NaFFPredictorTrainer
from codes.infonce_spatio_temporal import InfoNCESpatioTemporalTrainer
import wandb
import pandas as pd

data = np.zeros((50, 10, 20000))
finalData = np.zeros((50, 1000, 1, 10, 20))

for p in range(50):
    filename = '../TimeSeries/TSDataCSV'+str(p)+'.csv'
    print(filename)
    df = pd.read_csv(filename)
    data[p, :, :] = df


for i in range(50):
    for j in range(1000):
        finalData[i, j, 0, :, :] = data[i, :, j*20:j*20+20]


print(finalData.shape)


def train_encoder(args):
    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")

    '''
    tr_eps, val_eps = get_episodes(steps=args.probe_steps,
                                 env_name=args.env_name,
                                 seed=args.seed,
                                 num_processes=args.num_processes,
                                 num_frame_stack=args.num_frame_stack,
                                 downsample=not args.no_downsample,
                                 color=args.color,
                                 entropy_threshold=args.entropy_threshold,
                                 collect_mode=args.probe_collect_mode,
                                 train_mode="train_encoder",
                                 checkpoint_index=args.checkpoint_index,
                                 min_episode_length=args.batch_size)
    

    observation_shape = tr_eps[0][0].shape
    if args.encoder_type == "Nature":
        encoder = NatureCNN(observation_shape[0], args)
    elif args.encoder_type == "Impala":
        encoder = ImpalaCNN(observation_shape[0], args)
    encoder.to(device)
    torch.set_num_threads(1)

    config = {}
    config.update(vars(args))
    config['obs_space'] = observation_shape  # weird hack

    if args.method == "infonce-stdim":
        trainer = InfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "global-infonce-stdim":
        trainer = GlobalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "global-local-infonce-stdim":
        trainer = GlobalLocalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "dim":
        trainer = DIMTrainer(encoder, config, device=device, wandb=wandb)
    else:
        assert False, "method {} has no trainer".format(args.method)

    trainer.train(tr_eps, val_eps)

    return encoder
'''


'''
if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['pretraining-only']
    wandb.init(project=args.wandb_proj, entity="curl-atari", tags=tags)
    config = {}
    config.update(vars(args))
    wandb.config.update(config)
    train_encoder(args)
'''