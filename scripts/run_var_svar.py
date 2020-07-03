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
from src.fine_tuning import FineTuning
from src.lstm import subLSTM
import wandb
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from aari.episodes import get_episodes


def train_encoder(args):
    currentDT = datetime.datetime.now()
    d1 = currentDT.strftime("%Y-%m-%d%H:%M:%S")
    dir = 'run-' + d1
    wdb = 'wandb'
    path = os.path.join(os.getcwd(), wdb)
    path = os.path.join(path, dir)
    args.path = path
    os.mkdir(path)
    sample_x = 10
    sample_y = 20
    subjects = 50
    tc = 20000
    samples_per_subject = int(tc / sample_y)
    samples_per_subject=200
    # samples_per_subject = int((tc - sample_y)+1)
    ntrain_samples = 35
    ntest_samples = 15
    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")

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
    data = np.zeros((subjects, sample_x, tc))
    finalData = np.zeros((subjects, samples_per_subject, sample_x, sample_y))
    sample_x = 53
    data2= np.zeros((subjects,sample_x,tc))
    finalData2 = np.zeros((subjects, samples_per_subject, sample_x, sample_y))
    for p in range(subjects):
        filename = '../VARandSVARTimeSeries/TSDataCSV' + str(p) + '.csv'
        print(filename)
        df = pd.read_csv(filename)
        data[p, :, :] = df
        #d = df.values
        #data[p, :, :] = d[:, 0:tc]

    # f=plt.figure()
    # plt.subplot(221)
    # for a in range(100):
    #     plt.plot(data[0,a,:])
    # plt.subplot(222)
    # for a in range(100):
    #     plt.plot(data[1,a,:])
    # plt.subplot(223)
    # for a in range(100):
    #     plt.plot(data[2, a, :])
    # plt.subplot(224)
    # for a in range(100):
    #     plt.plot(data[3, a, :])
    # plt.show()
    # f.savefig("TS.pdf", bbox_inches='tight')
    for i in range(subjects):
        for j in range(samples_per_subject):
            print('a')
            # finalData[i, j, :, :] = data[i, :, j :j  + sample_y]
            finalData[i, j, :, :] = data[i, :, (j * 20):(j * 20) + sample_y]

    print(finalData.shape)




    filename = 'correct_indices.csv'
    print(filename)
    df = pd.read_csv(filename, header=None)
    c_indices = df.values
    c_indices = torch.from_numpy(c_indices).int()
    c_indices = c_indices.view(53)
#    finalData2 = finalData[:,:,c_indices,:]





    filename = 'index_array_var.csv'
    df = pd.read_csv(filename, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).int()
    index_array = index_array.view(subjects)

    filename = 'labels_VARsVAR.csv'
    df = pd.read_csv(filename, header=None)
    all_labels = df.values
    all_labels = torch.from_numpy(all_labels).int()
    all_labels = all_labels.view(subjects)
    #all_labels = all_labels - 1
    # all_labels = torch.randint(high=2, size=(311,), dtype=torch.int64)
    ID = args.script_ID - 1
    ID = ID * ntest_samples
    # val_indexs = ID-1;
    # val_indexe = ID+200

    test_indexs = ID
    test_indexe = ID + ntest_samples

    tr_index1 = index_array[0:20]
    tr_index2 = index_array[25:45]

    tr_index = torch.cat((tr_index1, tr_index2))

    test_index1 = index_array[20:25]
    test_index2 = index_array[45:50]
    test_index = torch.cat((test_index1, test_index2))

    # print(tr_index)
    tr_eps = finalData[tr_index, :, :, :]
    val_eps = finalData[0:10, :, :, :]
    test_eps = finalData[test_index, :, :, :]

    tr_labels = all_labels[tr_index.long()]

    test_labels = all_labels[test_index.long()]
    tr_eps = torch.from_numpy(tr_eps).float()
    val_eps = torch.from_numpy(val_eps).float()
    test_eps = torch.from_numpy(test_eps).float()

    tr_eps.to(device)
    val_eps.to(device)
    test_eps.to(device)

    # print("index_arrayshape", index_array.shape)
    print("trainershape", tr_eps.shape)
    print("valshape", val_eps.shape)
    print("testshape", test_eps.shape)
    print('ID = ', args.script_ID)

    print(tr_labels)
    print(test_labels)
    observation_shape = finalData.shape
    if args.encoder_type == "Nature":
        encoder = NatureCNN(observation_shape[2], args)
        # encoder = NatureCNN(input_channels, args)

    elif args.encoder_type == "Impala":
        encoder = ImpalaCNN(observation_shape[2], args)

    elif args.encoder_type == "NatureOne":
        encoder = NatureOneCNN(observation_shape[2], args)
        model_dict = torch.load(
            '/Users/umahmood1/Documents/ST_DIM/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/scripts/wandb/encoder.pt')
        # model_dict = torch.load('/Users/umahmood1/Documents/ST_DIM/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/scripts/wandb//run-2019-09-0614:11:16/encoder.pt',map_location=torch.device('cpu'))
        # model_dict = torch.load('/Users/umahmood1/Documents/ST_DIM/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/scripts/wandb/run-2019-09-0414:51:25/MontezumaRevengeNoFrameskip-v4.pt',map_location=torch.device('cpu'))
    # print(encoder)
    encoder.load_state_dict(model_dict)
    # encoder.eval()
    encoder.to(device)
    torch.set_num_threads(1)

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
        trainer = FineTuning(encoder, config, device=device, tr_labels=tr_labels, test_labels=test_labels, wandb=wandb)
    elif args.method == "sub-lstm":
        trainer = subLSTM(encoder, config, device=device, tr_labels=tr_labels, test_labels=test_labels, embedding_dim=256, hidden_dim=200, num_layers=1,  wandb=wandb)
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
    # wandb.init(project=args.wandb_proj, entity="curl-atari", tags=tags)
    config = {}
    config.update(vars(args))
    # wandb.config.update(config)
    train_encoder(args)
