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
from src.encoders_LS import NatureCNN, ImpalaCNN, NatureOneCNN
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
import h5py
# from tensorboardX import SummaryWriter


def find_indices_of_each_class(all_labels):
    HC_index = (all_labels == 0).nonzero()
    SZ_index = (all_labels == 1).nonzero()

    return HC_index, SZ_index


def train_encoder(args):
    start_time = time.time()
    # do stuff

    # ID = args.script_ID + 3
    ID = args.script_ID - 1
    JobID = args.job_ID


    # ID = 4
    sID = str(ID)
    currentDT = datetime.datetime.now()
    d1 = currentDT.strftime("%Y-%m-%d%H:%M:%S")
    d1 = str(JobID) + '_' + str(ID) + '_' + d1
    Name = 'NPT_KW'
    dir = 'run-' + d1 + Name
    dir = dir + '-' + str(ID)
    wdb = 'wandb_new'

    wpath = os.path.join(os.getcwd(), wdb)
    path = os.path.join(wpath, dir)
    args.path = path
    os.mkdir(path)

    wdb1 = 'wandb'
    wpath1 = os.path.join(os.getcwd(), wdb1)

    fig = 'Fig'
    fig_path = os.path.join(os.getcwd(), fig)
    fig_path = os.path.join(fig_path, dir)
    args.fig_path = fig_path
    os.mkdir(fig_path)

    p = 'UF'
    dir = 'run-2019-09-1223:36:31' + '-' + str(ID) + 'FPT_ICA_COBRE'
    p_path = os.path.join(os.getcwd(), p)
    p_path = os.path.join(p_path, dir)
    args.p_path = p_path
    # os.mkdir(fig_path)
    # hf = h5py.File('../FBIRN_AllData.h5', 'w')
    tfilename = str(JobID) + 'outputFILE' + Name + str(ID)

    output_path = os.path.join(os.getcwd(), 'Output')
    output_path = os.path.join(output_path, tfilename)
    # output_text_file = open(output_path, "w+")
    # writer = SummaryWriter('exp-1')
    ntrials = 10
    ngtrials = 10
    best_auc = 0.
    best_gain = 0
    current_gain=0
    tr_sub = [15, 25, 50, 75, 100]

    # With 16 per sub val, 10 WS
    # gain = [0.2, 0.7, 0.95, 0.2, 0.7]     #FPT
    # gain = [0.35, 0.85, 0.6, 0.55, 0.25]    #UFPT
    # gain = [0.3, 0.55, 0.8, 0.6, 0.85]  # NPT

    # With 32 per sub val, 10 WS
    # gain = [0.4, 0.2, 0.25, 0.3, 0.3]     #FPT
    # gain = [0.1, 0.25, 0.5, 0.35, 0.45]    #UFPTr
    # gain = [0.85, 0.35, 0.70, 0.4, 0.75]  # NPTr

    # With 16 per sub val, and 20 Window Shift
    # gain = [0.45, 0.35, 0.25, 0.65, 0.55]     #FPTr
    # gain = [0.15, 0.25, 0.45, 0.5, 0.1]    #UFPTr
    # gain = [0.95, 0.35, 0.8, 0.85, 0.45]  # NPTr

    # With 16 per sub val, 10 WS, super encoder
    # gain = [0.85, 0.5, 0.7, 0.8, 0.55]     #FPT
    # gain = [0.4, 0.4, 0.3, 0.1, 0.95]    #UFPT
    # gain = [0.65, 0.35, 0.3, 0.25, 0.85]  # NPT

    gain = [0.5, 0.5, 0.5, 0.5, 0.5]  # NPT
    sub_per_class = tr_sub[ID]
    current_gain = gain[ID]
    sample_x = 128
    sample_y = 5
    subjects = 3030
    tc = 87

    samples_per_subject = int(tc / sample_y)
    samples_per_subject = 18
    # samples_per_subject = int((tc - sample_y)+1)
    ntest_samples_perclass = 64
    nval_samples_perclass = 64
    window_shift = 5

    if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        device = torch.device("cuda:" + cudaID)
        # device = torch.device("cuda:" + str(args.cuda_id))
    else:
        device = torch.device("cpu")
    print('device = ', device)
    # return
    # For ICA TC Data
    data = np.zeros((subjects, sample_x, tc))
    finalData = np.zeros((subjects, samples_per_subject, sample_x, sample_y))
    n_good_comp = 128
    finalData2 = np.zeros((subjects, samples_per_subject, n_good_comp, sample_y))

    # for p in range(subjects):
    #     filename = '../TC_label_FBIRN/FBIRN_ica_br' + str(p + 1) + '.csv'
    #     if p % 30 == 0:
    #         print(filename)
    #     df = pd.read_csv(filename, header=None)
    #     d = df.values
    #     data[p, :, :] = d[:, 0:tc]

    # data2 = data.reshape(subjects, sample_x * tc)
    # hf.create_dataset('FBIRN_dataset', data=data2)
    # hf.close()
    hf = h5py.File('../Cat_NoCat_AllData.h5', 'r')
    data2 = hf.get('Cat_NoCat')
    data2 = np.array(data2)
    data = data2.reshape((subjects,sample_x,tc))

    # for p in range(subjects):
    #     for q in range(sample_x):
    #         for r in range(tc):
    #             diff = data[p, q, r] - data2[p, q, r];
    #             print(diff);

    for i in range(subjects):
        for j in range(samples_per_subject):
            if (j!=samples_per_subject-1):
                finalData[i, j, :, :] = data[i, :, (j * window_shift):(j * window_shift) + sample_y]
            else:
                finalData[i, j, :, 0:2] = data[i, :, (j * window_shift):(j * window_shift) + 2]
                finalData[i, j, :, 2:4] = 0.


    print(finalData.shape)


    finalData2 = finalData.copy()

    filename = 'index_array_labelled_KW2.csv'
    df = pd.read_csv(filename, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).long()
    index_array = index_array.view(subjects)

    all_labels = np.zeros((3030))
    all_labels[:1515] = 1.
    all_labels = torch.from_numpy(all_labels)
    all_labels = all_labels.view(subjects)

    finalData2 = finalData2[index_array, :, :, :]
    all_labels = all_labels[index_array]
    # all_labels = torch.randint(high=2, size=(311,), dtype=torch.int64)

    HC_index, SZ_index = find_indices_of_each_class(all_labels)


    cross_validation = 20
    results = torch.zeros(ntrials*cross_validation, 4)
    results_index=0
    for cv in range(cross_validation):

        test_start_index = cv * ntest_samples_perclass
        test_end_index = test_start_index + ntest_samples_perclass
        print(test_start_index)
        print(test_end_index)
        total_HC_index_tr_val = torch.cat([HC_index[:test_start_index], HC_index[test_end_index:]])
        total_SZ_index_tr_val = torch.cat([SZ_index[:test_start_index], SZ_index[test_end_index:]])

        HC_index_test = HC_index[test_start_index:test_end_index]
        SZ_index_test = SZ_index[test_start_index:test_end_index]

        total_HC_index_tr = total_HC_index_tr_val[:(len(total_HC_index_tr_val) - nval_samples_perclass)]
        total_SZ_index_tr = total_SZ_index_tr_val[:(len(total_SZ_index_tr_val) - nval_samples_perclass)]

        HC_index_val = total_HC_index_tr_val[(len(total_HC_index_tr_val) - nval_samples_perclass):]
        SZ_index_val = total_SZ_index_tr_val[(len(total_SZ_index_tr_val) - nval_samples_perclass):]


        auc_arr = torch.zeros(ngtrials, 1)
        avg_auc = 0.
        for trial in range(ntrials):
                print ('trial = ', trial);
                output_text_file = open(output_path, "a+")
                output_text_file.write("Trial = %d\r\n" % (trial))
                output_text_file.close()
                # writer.add_scalar('trial', trial)
                #current_gain = (trial+1) * 0.05
                #for g_trial in range (ngtrials):
                # Get subject_per_class number of random values
                HC_random = torch.randperm(len(total_HC_index_tr));
                SZ_random = torch.randperm(len(total_SZ_index_tr));
                HC_random = HC_random[:sub_per_class]
                SZ_random = SZ_random[:sub_per_class]

                # HC_random = torch.randint(high=len(total_HC_index_tr), size=(sub_per_class,))
                # SZ_random = torch.randint(high=len(total_SZ_index_tr), size=(sub_per_class,))
                #

                # Choose the subject_per_class indices from HC_index_val and SZ_index_val using random numbers

                HC_index_tr = total_HC_index_tr[HC_random]
                SZ_index_tr = total_SZ_index_tr[SZ_random]

                # ID = ID * ntest_samples
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

                tr_labels = tr_labels.to(device)
                val_labels = val_labels.to(device)
                test_labels = test_labels.to(device)
                # print('tr1', tr_eps.device, tr_eps.dtype, type(tr_eps), tr_eps.type())
                tr_eps = tr_eps.to(device)
                val_eps = val_eps.to(device)
                test_eps = test_eps.to(device)
                # asdfasdf = tr_eps.to(device)
                # print('tr2', tr_eps.device, tr_eps.dtype, type(tr_eps), tr_eps.type())
                # print('asdf',asdfasdf.device, asdfasdf.dtype, type(asdfasdf), asdfasdf.type())
                # return;
                # print("index_arrayshape", index_array.shape)
                #     print("trainershape", tr_eps.shape)
                #     print("valshape", val_eps.shape)
                #     print("testshape", test_eps.shape)
                #     print('ID = ', args.script_ID)

                # print(tr_labels)
                # print(test_labels)
                observation_shape = finalData2.shape
                if args.encoder_type == "Nature":
                    encoder = NatureCNN(observation_shape[2], args)
                # encoder = NatureCNN(input_channels, args)

                elif args.encoder_type == "Impala":
                    encoder = ImpalaCNN(observation_shape[2], args)

                elif args.encoder_type == "NatureOne":
                    # encoder = NatureOneCNN(observation_shape[2], args)
                    dir = 'run-2019-09-1018:25:39/encoderLS.pt'
                    path = os.path.join(wpath1, dir)
                    encoder = NatureOneCNN(observation_shape[2], args)
                    lstm_model = subjLSTM(device, args.feature_size, args.lstm_size, num_layers=args.lstm_layers,
                                          freeze_embeddings=True, gain=current_gain)
                    # model_dict = torch.load(
                    #     '/Users/umahmood1/Documents/ST_DIM/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/scripts/wandb//run-2019-09-1009:09:14/encoder.pt',
                    #     map_location=torch.device('cpu'))  # with all components
                    # model_dict = torch.load(path, map_location=device)  # with good components
                    # model_dict = torch.load('/Users/umahmood1/Documents/ST_DIM/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/scripts/wandb//run-2019-09-0614:11:16/encoder.pt',map_location=torch.device('cpu'))
                    # model_dict = torch.load('/Users/umahmood1/Documents/ST_DIM/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/scripts/wandb/run-2019-09-0414:51:25/MontezumaRevengeNoFrameskip-v4.pt',map_location=torch.device('cpu'))
                # print(encoder)
                # encoder.load_state_dict(model_dict)
                encoder.to(device)
                lstm_model.to(device)
                #torch.set_num_threads(1)

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
                                          val_labels=val_labels, test_labels=test_labels, wandb=wandb, trial=str(trial), crossv=str(cv))
                elif args.method == "global-infonce-stdim":
                    trainer = GlobalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
                elif args.method == "global-local-infonce-stdim":
                    trainer = GlobalLocalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
                elif args.method == "dim":
                    trainer = DIMTrainer(encoder, config, device=device, wandb=wandb)
                else:
                    assert False, "method {} has no trainer".format(args.method)

                results[results_index][0], results[results_index][1], results[results_index][2], results[results_index][3] = trainer.train(tr_eps, val_eps, test_eps)
                results_index = results_index + 1
                #auc_arr[g_trial] = trainer.train(tr_eps, val_eps, test_eps)
                # return

        #avg_auc = auc_arr.mean()
        #if avg_auc > best_auc:
        #    best_auc = avg_auc
        #    best_gain = current_gain
    np_results = results.numpy()
    tresult_csv = os.path.join(args.path, 'test_results' + sID + '.csv')
    np.savetxt(tresult_csv, np_results, delimiter=",")
    #
    # return encoder
    #print ('best gain = ', best_gain)
    elapsed = time.time() - start_time
    print('total time = ', elapsed);


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['pretraining-only']
    config = {}
    config.update(vars(args))
    train_encoder(args)
