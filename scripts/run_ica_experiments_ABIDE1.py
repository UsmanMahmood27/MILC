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
from src.slstm_attn import LSTMTrainer
from src.lstm_attn import subjLSTM
import wandb
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from aari.episodes import get_episodes
import h5py


def find_indices_of_each_class(all_labels):
    HC_index = (all_labels == 0).nonzero()
    SZ_index = (all_labels == 1).nonzero()

    return HC_index, SZ_index


def train_encoder(args):
    ID = args.script_ID - 1
    JobID = args.job_ID

    ID = 5
    print('ID = ' + str(ID))
    print('exp = ' + args.exp)
    print('pretraining = ' + args.pre_training)
    sID = str(ID)
    currentDT = datetime.datetime.now()
    d1 = currentDT.strftime("%Y-%m-%d%H:%M:%S")
    d2 = str(JobID) + '_' + str(ID)

    Name = args.exp + '_ABIDE' + args.pre_training + 'after_lstm'
    dir = 'run-' + d1 + d2 + Name
    dir = dir + '-' + str(ID)
    wdb = 'wandb_new'

    wpath = os.path.join(os.getcwd(), wdb)
    path = os.path.join(wpath, dir)
    args.path = path
    os.mkdir(path)

    wdb1 = 'wandb_new'
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
    # hf = h5py.File('../ABIDE1_AllData.h5', 'w')
    ntrials = 10
    ngtrials = 10
    best_auc=0.
    best_gain=0
    # current_gain=0
    tr_sub = [15, 25, 50, 75, 100, 150]
    if args.exp == 'FPT':
        gain = [0.55, 0.95, 0.25, 0.5, 0.5, 0.3]     #FPT
    elif args.exp == 'UFPT':
        gain = [0.15, 0.2, 0.25, 0.2, 0.35, 0.35]    #UFPT
    else:
        gain = [0.4, 0.5, 0.75, 0.75, 0.45, 0.8]     #NPT


    tfilename = str(JobID) + 'outputFILE' + Name + str(ID)

    output_path = os.path.join(os.getcwd(), 'Output')
    output_path = os.path.join(output_path, tfilename)

    sub_per_class = tr_sub[ID]
    current_gain = gain[ID]
    sample_x = 100
    sample_y = 20
    subjects = 569
    tc = 140
    samples_per_subject = int(tc / sample_y)
    samples_per_subject = 13
    # samples_per_subject = int((tc - sample_y)+1)
    ntest_samples_perclass = 50
    nval_samples_perclass = 50
    test_start_index = 0
    test_end_index = ntest_samples_perclass
    window_shift = 10

    if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        device = torch.device("cuda:" + cudaID)
        # device = torch.device("cuda:" + str(args.cuda_id))
    else:
        device = torch.device("cpu")
    print('device = ', device)

    # For ICA TC Data
    data = np.zeros((subjects, sample_x, tc))
    finalData = np.zeros((subjects, samples_per_subject, sample_x, sample_y))
    n_good_comp = 53
    finalData2 = np.zeros((subjects, samples_per_subject, n_good_comp, sample_y))

    # for p in range(subjects):
    #     filename = '../TC_label_ABIDE1/ABIDE1_ica_br' + str(p + 1) + '.csv'
    #     if p % 30 == 0:
    #         print(filename)
    #     df = pd.read_csv(filename, header=None)
    #     d = df.values
    #     data[p, :, :] = d[:, 0:tc]

    # data2=data.reshape(subjects,sample_x*tc)
    # df = pd.read_csv('../COBRE_AllData.csv', header=None)
    # d = df.values
    # data = d
    # hf.create_dataset('ABIDE1_dataset', data=data2)
    # hf.close()
    # img = nib.load('../fMRI_Data/SM1.nii')
    hf = h5py.File('../Data/ABIDE1_AllData.h5', 'r')
    data = hf.get('ABIDE1_dataset')
    data = np.array(data)
    data = data.reshape(subjects, sample_x, tc)

    # Get Training indices for ABIDE and convert them to tensor. this is to have same training samples everytime.
    hf_hc = h5py.File('../IndicesAndLabels/ABIDE1_HC_TrainingIndex.h5', 'r')
    HC_TrainingIndex = hf_hc.get('HC_TrainingIndex')
    HC_TrainingIndex = np.array(HC_TrainingIndex)
    HC_TrainingIndex = torch.from_numpy(HC_TrainingIndex)

    hf_sz = h5py.File('../IndicesAndLabels/ABIDE1_SZ_TrainingIndex.h5', 'r')
    SZ_TrainingIndex = hf_sz.get('SZ_TrainingIndex')
    SZ_TrainingIndex = np.array(SZ_TrainingIndex)
    SZ_TrainingIndex = torch.from_numpy(SZ_TrainingIndex)


    for i in range(subjects):
        for j in range(samples_per_subject):
            finalData[i, j, :, :] = data[i, :, (j * window_shift):(j * window_shift) + sample_y]

    print(finalData.shape)
    filename = '../IndicesAndLabels/correct_indices_GSP.csv'
    print(filename)
    df = pd.read_csv(filename, header=None)
    c_indices = df.values
    c_indices = torch.from_numpy(c_indices).int()
    c_indices = c_indices.view(53)
    c_indices = c_indices - 1
    finalData2 = finalData[:, :, c_indices, :]

    filename = '../IndicesAndLabels/index_array_labelled_ABIDE1.csv'
    df = pd.read_csv(filename, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).long()
    index_array = index_array.view(subjects)

    filename = '../IndicesAndLabels/labels_ABIDE1.csv'
    df = pd.read_csv(filename, header=None)
    all_labels = df.values
    all_labels = torch.from_numpy(all_labels).int()
    all_labels = all_labels.view(subjects)
    all_labels = all_labels - 1
    finalData2 = finalData2[index_array, :, :, :]
    all_labels = all_labels[index_array]
    # all_labels = torch.randint(high=2, size=(311,), dtype=torch.int64)

    HC_index, SZ_index = find_indices_of_each_class(all_labels)

    total_HC_index_tr_val = torch.cat([HC_index[:test_start_index], HC_index[test_end_index:]])
    total_SZ_index_tr_val = torch.cat([SZ_index[:test_start_index], SZ_index[test_end_index:]])

    HC_index_test = HC_index[test_start_index:test_end_index]
    SZ_index_test = SZ_index[test_start_index:test_end_index]

    total_HC_index_tr = total_HC_index_tr_val[:(total_HC_index_tr_val.shape[0] - nval_samples_perclass)]
    total_SZ_index_tr = total_SZ_index_tr_val[:(total_SZ_index_tr_val.shape[0] - nval_samples_perclass)]

    HC_index_val = total_HC_index_tr_val[(total_HC_index_tr_val.shape[0] - nval_samples_perclass):]
    SZ_index_val = total_SZ_index_tr_val[(total_SZ_index_tr_val.shape[0] - nval_samples_perclass):]

    results = torch.zeros(ntrials, 4)
    auc_arr = torch.zeros(ngtrials, 1)
    avg_auc=0.
    HCrand=[]
    SZrand=[]
    for trial in range(ntrials):
            print('trial = ', trial);
            output_text_file = open(output_path, "a+")
            output_text_file.write("Trial = %d\r\n" % (trial))
            output_text_file.close()

        # current_gain = (trial+1) * 0.05
        # for g_trial in range (ngtrials):
        # Get subject_per_class number of random values
        #     a = torch.randperm(total_HC_index_tr.shape[0])
        #     print(a)
        #     HCrand.append(torch.randperm(total_HC_index_tr.shape[0]))
        #     SZrand.append(torch.randperm(total_SZ_index_tr.shape[0]))

            HC_random = HC_TrainingIndex[trial]
            SZ_random = SZ_TrainingIndex[trial]
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

            tr_eps.to(device)
            val_eps.to(device)
            test_eps.to(device)

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
            # encoder = NatureCNN(input_channels, args) as

            elif args.encoder_type == "Impala":
                encoder = ImpalaCNN(observation_shape[2], args)


            elif args.encoder_type == "NatureOne":

                # encoder = NatureOneCNN(observation_shape[2], args)

                dir = ""

                if args.pre_training == "basic":

                    dir = 'PreTrainedEncoders/Basic_Encoder/encoder.pt'

                elif args.pre_training == "milc":

                    dir = 'PreTrainedEncoders/Milc/encoder.pt'

                    args.oldpath = wpath1 + '/PreTrainedEncoders/Milc'

                elif args.pre_training == "two-loss-milc":

                    dir = 'PreTrainedEncoders/Milc_two_loss_after_lstm/encoder.pt'

                    args.oldpath = wpath1 + '/PreTrainedEncoders/Milc_two_loss_after_lstm'

                elif args.pre_training == "variable-attention":

                    dir = 'PreTrainedEncoders/VariableAttention/encoder.pt'

                    args.oldpath = wpath1 + '/PreTrainedEncoders/VariableAttention'

                path = os.path.join(wpath1, dir)

                # dir = 'PreTrainedEncoders/VariableAttention/encoder.pt'

                # args.oldpath = wpath1 + '/run-2020-01-0401:25:58'

                # path = os.path.join(wpath1, dir)

                encoder = NatureOneCNN(observation_shape[2], args)

                lstm_model = subjLSTM(device, args.feature_size, args.lstm_size, num_layers=args.lstm_layers,

                                      freeze_embeddings=True, gain=current_gain)

                # model_dict = torch.load(

                #     '/Users/umahmood1/Documents/ST_DIM/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/scripts/wandb//run-2019-09-1009:09:14/encoder.pt',

                #     map_location=torch.device('cpu'))  # with all components

                model_dict = torch.load(path, map_location=device)  # with good components

                # model_dict = torch.load('/Users/umahmood1/Documents/ST_DIM/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/scripts/wandb//run-2019-09-0614:11:16/encoder.pt',map_location=torch.device('cpu'))

                # model_dict = torch.load('/Users/umahmood1/Documents/ST_DIM/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/scripts/wandb/run-2019-09-0414:51:25/MontezumaRevengeNoFrameskip-v4.pt',map_location=torch.device('cpu'))

            # print(encoder)

            if args.exp in ['UFPT', 'FPT']:
                encoder.load_state_dict(model_dict)

            encoder.to(device)

            lstm_model.to(device)

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
                                    val_labels=val_labels, test_labels=test_labels, wandb=wandb, trial=str(trial))
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
            # auc_arr[g_trial] = trainer.train(tr_eps, val_eps, test_eps)

        # avg_auc = auc_arr.mean()
        # if avg_auc > best_auc:
        #     best_auc = avg_auc
        #     best_gain = current_gain
    np_results = results.numpy()
    tresult_csv = os.path.join(args.path, 'test_results' + sID + '.csv')
    np.savetxt(tresult_csv, np_results, delimiter=",")

    # hfn = h5py.File('../ABIDE1_HC_TrainingIndex.h5', 'w')
    # hfn.create_dataset('HC_TrainingIndex', data=torch.stack(HCrand))
    # hfn.close()
    #
    # hfq = h5py.File('../ABIDE1_SZ_TrainingIndex.h5', 'w')
    # hfq.create_dataset('SZ_TrainingIndex', data=torch.stack(SZrand))
    # hfq.close()



    # return encoder
    # print ('best gain = ', best_gain)


if __name__ == "__main__":
    torch.manual_seed(5)
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['pretraining-only']
    config = {}
    config.update(vars(args))
    train_encoder(args)
