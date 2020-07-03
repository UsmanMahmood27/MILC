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
from src.encoders_ICA import NatureCNN, ImpalaCNN, NatureOneCNN
from src.cpc import CPCTrainer
from src.vae import VAETrainer
from src.no_action_feedforward_predictor import NaFFPredictorTrainer
from src.infonce_spatio_temporal import InfoNCESpatioTemporalTrainer
from src.fine_tuning import FineTuning
import wandb
import pandas as pd
import datetime
from aari.episodes import get_episodes
import h5py

def train_encoder(args):
    currentDT = datetime.datetime.now()
    d1 = currentDT.strftime("%Y-%m-%d%H:%M:%S")
    dir = 'run-' + d1 + 'OASISSuper'
    wdb='wandb'
    wpath = os.path.join(os.getcwd(), wdb)
    path = os.path.join(wpath, dir)
    args.path = path
    os.mkdir(path)
    if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        device = torch.device("cuda:" + cudaID)
        # device = torch.device("cuda:" + str(args.cuda_id))
    else:
        device = torch.device("cpu")
    # device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")

    # tr_eps, val_eps = get_episodes(steps=args.probe_steps,
    #                              env_name=args.env_name,
    #                              seed=args.seed,
    #                              num_processes=args.num_processes,
    #                              num_frame_stack=args.num_frame_stack,
    #                              downsample=not args.no_downsample,
    #                              color=args.color,
    #                              entropy_threshold=args.entropy_threshold,
    #                              collect_mode=args.probe_collect_mode,
    #                              train_mode="train_encoder",
    #                              checkpoint_index=args.checkpoint_index,
    #                              min_episode_length=args.batch_size)

    # for Simulated TS
    # data = np.zeros((50, 10, 20000))
    # finalData = np.zeros((50, 1000, 10, 20))
    #
    # for p in range(50):
    #     filename = '../TimeSeries/TSDataCSV' + str(p) + '.csv'
    #     # filename = '../TimeSeries/HCP_ica_br1.csv'
    #     print(filename)
    #     df = pd.read_csv(filename)
    #     # print('size is',df.shape)
    #     # print(df[1,:])
    #     data[p, :, :] = df
    #
    # for i in range(50):
    #     for j in range(1000):
    #         finalData[i, j, :, :] = data[i, :, j * 20:j * 20 + 20]
    #
    # print(finalData.shape)
    #
    # tr_eps = finalData[:, 0:700, :, :]
    # val_eps = finalData[:, 700:900, :, :]
    # test_eps = finalData[:, 900:1000, :, :]

    # For ICA TC Data
    data = np.zeros((372, 100, 120))
    # modifieddata = np.zeros((823, 100, 1100))
    finalData = np.zeros((372, 11, 100, 20))
    finalData2 = np.zeros((372, 11, 53, 20))

    # for p in range(311):
    #     # print(p)
    #     filename = '../TC_label_FBIRN/FBIRN_ica_br' + str(p + 1) + '.csv'
    #     # filename = '../TimeSeries/HCP_ica_br1.csv'
    #     print(filename)
    #     df = pd.read_csv(filename, header=None)
    #
    #     # print('size is',df.shape)
    #     # print(df[1,:])
    #     # print('shape is', df.shape)
    #     d = df.values
    #     data[p, :, :] = d[:, 0:140]
        # modifieddata[p, :, :]= data[p, :, 0:1100]
    # print(data.shape)
    hf = h5py.File('../OASIS3_AllData.h5', 'r')
    data2 = hf.get('OASIS3_dataset')
    data2 = np.array(data2)
    data2 = data2.reshape((372, 100, 120))
    data = data2

    for i in range(372):
        for j in range(11):
            finalData[i, j, :, :] = data[i, :, (j * 10):(j * 10) + 20]

    print(finalData.shape)
    filename = 'index_array_labelled_OASIS3.csv'
    df = pd.read_csv(filename, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).int()
    index_array = index_array.view(372)

    filename = 'correct_indices_GSP.csv'
    print(filename)
    df = pd.read_csv(filename, header=None)
    c_indices = df.values
    c_indices = torch.from_numpy(c_indices).int()
    c_indices = c_indices.view(53)
    c_indices = c_indices - 1
    finalData2 = finalData[:, :, c_indices, :]
    # print(c_indices)

    # index_array = torch.randperm(823)
    # index_array = torch.randperm(823)
    # tr_index = index_array[0:500]
    # val_index = index_array[0:500]
    # test_index = index_array[0:500]

    ID = args.script_ID - 1
    ID = ID * 100
    # val_indexs = ID-1;
    # val_indexe = ID+200

    test_indexs = ID
    test_indexe = ID + 100

    tr_index1 = index_array[0:test_indexs]
    tr_index2 = index_array[test_indexe:372]

    tr_index = torch.cat((tr_index1, tr_index2))

    test_index = index_array[test_indexs:test_indexe]
    # print(tr_index)
    tr_eps = finalData2[tr_index, :, :, :]
    val_eps = finalData2[500:700, :, :, :]
    test_eps = finalData2[test_index, :, :, :]

    tr_eps = torch.from_numpy(tr_eps).float()
    val_eps = torch.from_numpy(val_eps).float()
    test_eps = torch.from_numpy(test_eps).float()
    input_data = torch.rand(20, 10, 1, 160, 210)
    # tr_eps = torch.rand(20, 10, 1, 160, 210)
    # val_eps = torch.rand(20, 10, 1, 160, 210)
    # observation_shape = tr_eps[0][0].shape

    tr_eps.to(device)
    val_eps.to(device)
    test_eps.to(device)

    print("index_arrayshape", index_array.shape)
    print("trainershape", tr_eps.shape)
    print("valshape", val_eps.shape)
    print("testshape", test_eps.shape)
    initial_n_channels = 1
    print('ID = ', args.script_ID)

    observation_shape = finalData2.shape
    if args.encoder_type == "Nature":
        encoder = NatureCNN(observation_shape[2], args)
        # encoder = NatureCNN(input_channels, args)

    elif args.encoder_type == "Impala":
        encoder = ImpalaCNN(observation_shape[2], args)

    elif args.encoder_type == "NatureOne":
        encoder = NatureOneCNN(observation_shape[2], args)
    dir = 'run-2019-09-1018:25:39/encoder.pt'
    ep = os.path.join(wpath, dir)
    model_dict = torch.load(ep, map_location=device)  # with good components
    encoder.load_state_dict(model_dict)
    encoder.to(device)
    # torch.set_num_threads(1)

    config = {}
    config.update(vars(args))
    # print("trainershape", os.path.join(wandb.run.dir, config['env_name'] + '.pt'))
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
    elif args.method == "fine-tuning":
        trainer = FineTuning(encoder, config, device=device, wandb=wandb)
    elif args.method == "global-infonce-stdim":
        trainer = GlobalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "global-local-infonce-stdim":
        trainer = GlobalLocalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "dim":
        trainer = DIMTrainer(encoder, config, device=device, wandb=wandb)
    else:
        assert False, "method {} has no trainer".format(args.method)

    # print("valshape", val_eps.shape)
    trainer.train(tr_eps, val_eps, test_eps)

    # return encoder


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['pretraining-only']
    # wandb.init()
    #wandb.init(project=args.wandb_proj, entity="curl-atari", tags=tags)
    config = {}
    config.update(vars(args))
    # wandb.config.update(config)
    train_encoder(args)
