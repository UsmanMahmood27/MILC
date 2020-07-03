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
from src.slstm import LSTMTrainer
from src.lstm import subjLSTM
import wandb
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from aari.episodes import get_episodes


def find_indices_of_each_class(all_labels):
    HC_index = (all_labels == 0).nonzero()
    SZ_index = (all_labels == 1).nonzero()

    return HC_index, SZ_index


def train_encoder(args):
    ID = args.script_ID - 1
    # ID = 0
    sID = str(ID)
    currentDT = datetime.datetime.now()
    d1 = currentDT.strftime("%Y-%m-%d%H:%M:%S")
    dir = 'run-' + d1 + 'FPT_No_EP_25'
    dir = dir + '-' + str(ID)
    wdb = 'wandb_new'

    wpath = os.path.join(os.getcwd(), wdb)
    path = os.path.join(wpath, dir)
    args.path = path
    os.mkdir(path)

    wdb1='wandb'
    wpath1 = os.path.join(os.getcwd(), wdb1)

    fig = 'Fig'
    fig_path = os.path.join(os.getcwd(), fig)
    fig_path = os.path.join(fig_path, dir)
    args.fig_path = fig_path
    os.mkdir(fig_path)

    p = 'UF'
    dir = 'run-2019-09-1223:36:31' + '-' + str(ID) + 'NPT_ICA'
    p_path = os.path.join(os.getcwd(), p)
    p_path = os.path.join(p_path, dir)
    args.p_path = p_path
    # os.mkdir(fig_path)

    ntrials = 10
    tr_sub = [15, 25, 50, 75, 100]
    sub_per_class = tr_sub[ID]
    sample_x = 100
    sample_y = 20
    subjects = 311
    subjects_for_train = sub_per_class * 2
    subjects_for_test = 64
    subjects_for_val = 47
    tc = 140
    samples_per_subject = int(tc / sample_y)
    samples_per_subject = 13
    # samples_per_subject = int((tc - sample_y)+1)
    ntrain_samples = 247
    ntest_samples = 64
    window_shift = 10
    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")

    # For ICA TC Data
    data = np.zeros((subjects, sample_x, tc))
    finalData = np.zeros((subjects, samples_per_subject, sample_x, sample_y))
    sample_x = 53
    finalData2 = np.zeros((subjects, samples_per_subject, sample_x, sample_y))

    for p in range(subjects):
        filename = '../TC_label_FBIRN/FBIRN_ica_br' + str(p + 1) + '.csv'
        print(filename)
        df = pd.read_csv(filename, header=None)
        # data[p, :, :] = df
        d = df.values
        data[p, :, :] = d[:, 0:tc]

    for i in range(subjects):
        for j in range(samples_per_subject):
            # print('a')
            # finalData[i, j, :, :] = data[i, :, j :j  + sample_y]
            finalData[i, j, :, :] = data[i, :, (j * window_shift):(j * window_shift) + sample_y]

    print(finalData.shape)

    filename = 'correct_indices_GSP.csv'
    print(filename)
    df = pd.read_csv(filename, header=None)
    c_indices = df.values
    c_indices = torch.from_numpy(c_indices).int()
    c_indices = c_indices.view(53)
    c_indices = c_indices - 1
    finalData2 = finalData[:, :, c_indices, :]

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

    HC_index, SZ_index = find_indices_of_each_class(all_labels)
    total_HC_index_tr = HC_index[:len(HC_index) - (32 + 16)]
    total_SZ_index_tr = SZ_index[:len(SZ_index) - (32 + 16)]

    # HC_index_val = HC_index[len(HC_index) - (32 + 16):len(HC_index) - 32]
    # SZ_index_val = SZ_index[len(HC_index) - (32 + 16):len(HC_index) - 32]

    HC_index_val = HC_index[len(HC_index) - (32 + 16):len(HC_index) - 32]
    SZ_index_val = SZ_index[len(HC_index) - (32 + 16):len(HC_index) - 32]

    HC_index_test = HC_index[len(HC_index) - (32):]
    SZ_index_test = SZ_index[len(SZ_index) - (32):]
    results = torch.zeros(ntrials, 4)
    auc = torch.zeros(ntrials, 1)

    for trial in range(ntrials):
        # Get subject_per_class number of random values
        HC_random = torch.randint(high=len(total_HC_index_tr), size=(sub_per_class,))
        SZ_random = torch.randint(high=len(total_SZ_index_tr), size=(sub_per_class,))
        #

        # Choose the subject_per_class indices from HC_index_val and SZ_index_val using random numbers

        HC_index_tr = total_HC_index_tr[HC_random]
        SZ_index_tr = total_SZ_index_tr[SZ_random]

        ID = ID * ntest_samples
        # val_indexs = ID-1;
        # val_indexe = ID+200

        tr_index = torch.cat((HC_index_tr, SZ_index_tr))
        val_index = torch.cat((HC_index_val, SZ_index_val))
        test_index = torch.cat((HC_index_test, SZ_index_test))

        tr_index = tr_index.view(tr_index.size(0))
        val_index = val_index.view(val_index.size(0))
        test_index = test_index.view(test_index.size(0))

        tr_eps = finalData2[tr_index, :, :, :]
        val_eps = finalData2[val_index, :, :, :]
        test_eps = finalData2[test_index, :, :, :]

        tr_labels = all_labels[tr_index.long()]
        val_labels = all_labels[val_index.long()]
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

        # print(tr_labels)
        # print(test_labels)
        observation_shape = finalData2.shape
        if args.encoder_type == "Nature":
            encoder = NatureCNN(observation_shape[2], args)
        # encoder = NatureCNN(input_channels, args)

        elif args.encoder_type == "Impala":
            encoder = ImpalaCNN(observation_shape[2], args)

        elif args.encoder_type == "NatureOne":
            encoder = NatureOneCNN(observation_shape[2], args)
            dir = 'run-2019-09-1018:25:39/encoder.pt'
            path = os.path.join(wpath1, dir)
            encoder = NatureOneCNN(observation_shape[2], args)
            lstm_model = subjLSTM(device, args.feature_size, args.lstm_size, num_layers=args.lstm_layers,
                                  freeze_embeddings=True)
            # model_dict = torch.load(
            #     '/Users/umahmood1/Documents/ST_DIM/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/scripts/wandb//run-2019-09-1009:09:14/encoder.pt',
            #     map_location=torch.device('cpu'))  # with all components
            model_dict = torch.load(path, map_location=device)  # with good components
            # model_dict = torch.load('/Users/umahmood1/Documents/ST_DIM/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/scripts/wandb//run-2019-09-0614:11:16/encoder.pt',map_location=torch.device('cpu'))
            # model_dict = torch.load('/Users/umahmood1/Documents/ST_DIM/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/scripts/wandb/run-2019-09-0414:51:25/MontezumaRevengeNoFrameskip-v4.pt',map_location=torch.device('cpu'))
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
            trainer = FineTuning(encoder, config, device=device, tr_labels=tr_labels, test_labels=test_labels,
                                 wandb=wandb)
        elif args.method == "sub-lstm":
            trainer = LSTMTrainer(encoder, lstm_model, config, device=device, tr_labels=tr_labels,
                                  val_labels=val_labels, test_labels=test_labels,
                                  wandb=wandb, trial=str(trial))
        elif args.method == "global-infonce-stdim":
            trainer = GlobalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
        elif args.method == "global-local-infonce-stdim":
            trainer = GlobalLocalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
        elif args.method == "dim":
            trainer = DIMTrainer(encoder, config, device=device, wandb=wandb)
        else:
            assert False, "method {} has no trainer".format(args.method)

        results[trial][0], results[trial][1], results[trial][2], results[trial][3] = trainer.train(tr_eps, val_eps,
                                                                                                   test_eps)
    np_results = results.numpy()
    tresult_csv = os.path.join(args.path, 'test_results' + sID + '.csv')
    np.savetxt(tresult_csv, np_results, delimiter=",")

    # return encoder


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['pretraining-only']
    config = {}
    config.update(vars(args))
    train_encoder(args)
