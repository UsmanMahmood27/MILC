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
from src.lstm import subLSTM
import wandb
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from aari.episodes import get_episodes




def train_encoder(args):
    ID = args.script_ID - 1
    ID = 2
    currentDT = datetime.datetime.now()
    d1 = currentDT.strftime("%Y-%m-%d%H:%M:%S")
    dir = 'run-' + d1
    dir = dir + '-' + str(ID)
    wdb = 'wandb'

    wpath = os.path.join(os.getcwd(), wdb)
    path = os.path.join(wpath, dir)
    args.path = path
    os.mkdir(path)

    fig = 'Fig'
    fig_path = os.path.join(os.getcwd(), fig)
    fig_path = os.path.join(fig_path, dir)
    args.fig_path = fig_path
    os.mkdir(fig_path)

    sample_x = 100
    sample_y = 20
    subjects = 311
    tc = 140
    samples_per_subject = int(tc / sample_y)
    samples_per_subject = 13
    # samples_per_subject=200
    # samples_per_subject = int((tc - sample_y)+1)
    ntrain_samples = 211
    ntest_samples = 100
    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")



    # For ICA TC Data
    data = np.zeros((subjects, sample_x, tc))
    finalData = np.zeros((subjects, samples_per_subject, sample_x, sample_y))
    sample_x = 53
    finalData2 = np.zeros((subjects, samples_per_subject, sample_x, sample_y))

    for p in range(subjects):
        filename = '../TC_label_FBIRN/FBIRN_ica_br' + str(p+1) + '.csv'
        print(filename)
        df = pd.read_csv(filename, header=None)
        # data[p, :, :] = df
        d = df.values
        data[p, :, :] = d[:, 0:tc]

    for i in range(subjects):
        for j in range(samples_per_subject):
            # print('a')
            # finalData[i, j, :, :] = data[i, :, j :j  + sample_y]
            finalData[i, j, :, :] = data[i, :, (j * 10):(j * 10) + sample_y]

    print(finalData.shape)




    filename = 'correct_indices_GSP.csv'
    print(filename)
    df = pd.read_csv(filename, header=None)
    c_indices = df.values
    c_indices = torch.from_numpy(c_indices).int()
    c_indices = c_indices.view(53)
    c_indices = c_indices - 1
    finalData2 = finalData[:,:,c_indices,:]
    # print(c_indices)




    filename = 'index_array_labelled_FBIRN2.csv'
    df = pd.read_csv(filename, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).int()
    index_array = index_array.view(subjects)

    filename = 'labels_FBIRN.csv'
    df = pd.read_csv(filename, header=None)
    all_labels = df.values
    all_labels = torch.from_numpy(all_labels).int()
    all_labels = all_labels.view(subjects)
    all_labels = all_labels - 1
    # all_labels = torch.randint(high=2, size=(311,), dtype=torch.int64)





    ID = ID * ntest_samples
    # val_indexs = ID-1;
    # val_indexe = ID+200

    test_indexs = ID
    test_indexe = ID + ntest_samples


    tr_index1 = index_array[0:test_indexs]
    tr_index2 = index_array[test_indexe:subjects]

    tr_index = torch.cat((tr_index1, tr_index2))

    test_index = index_array[test_indexs:test_indexe]

    tr_eps = finalData2[tr_index, :, :, :]
    val_eps = finalData2[0:10, :, :, :]
    test_eps = finalData2[test_index, :, :, :]

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
    observation_shape = finalData2.shape
    if args.encoder_type == "Nature":
        encoder = NatureCNN(observation_shape[2], args)
        # encoder = NatureCNN(input_channels, args)

    elif args.encoder_type == "Impala":
        encoder = ImpalaCNN(observation_shape[2], args)

    elif args.encoder_type == "NatureOne":
        dir = 'run-2019-09-1018:25:39/encoder.pt'
        path = os.path.join(wpath, dir)
        encoder = NatureOneCNN(observation_shape[2], args)
        # model_dict = torch.load(
        #     '/Users/umahmood1/Documents/ST_DIM/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/scripts/wandb//run-2019-09-1009:09:14/encoder.pt',
        #     map_location=torch.device('cpu'))  # with all components
        model_dict = torch.load(path,map_location=device) # with good components
    # print(encoder)
    encoder.load_state_dict(model_dict)
    encoder.eval()
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
        trainer = subLSTM(encoder, config, device=device, tr_labels=tr_labels, test_labels=test_labels, embedding_dim=256, hidden_dim=200, num_layers=1,  wandb=wandb, ID=ID)
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
    # torch.manual_seed(10)
    CURRENTSEED = torch.initial_seed()
    # Numpy expects unsigned integer seeds.
    np_seed = CURRENTSEED // (2 ** 32 - 1)
    np.random.seed(np_seed)

    parser = get_argparser()
    args = parser.parse_args()
    tags = ['pretraining-only']
    # wandb.init()
    # wandb.init(project=args.wandb_proj, entity="curl-atari", tags=tags)
    config = {}
    config.update(vars(args))
    # wandb.config.update(config)
    train_encoder(args)
