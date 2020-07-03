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
from src.slstm_attn_catalyst import LSTMTrainer
from src.lstm_attn import subjLSTM
from src.All_Architecture import combinedModel
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

    print('ID = ' + str(ID))
    print('exp = ' + args.exp)
    print('pretraining = ' + args.pre_training)
    sID = str(ID)
    currentDT = datetime.datetime.now()
    d1 = currentDT.strftime("%Y-%m-%d%H:%M:%S")
    d2 = str(JobID) + '_' + str(ID)

    Name = args.exp + '_FBIRN_' + args.pre_training + 'after_lstm'
    dir = 'run-' + d1 + d2 + Name
    dir = dir + '-' + str(ID)
    wdb = 'wandb_new'

    wpath = os.path.join(os.getcwd(), wdb)
    path = os.path.join(wpath, dir)
    args.path = path
    #os.mkdir(path)

    wdb1 = 'wandb_new'
    wpath1 = os.path.join(os.getcwd(), wdb1)

    fig = 'Fig'
    fig_path = os.path.join(os.getcwd(), fig)
    fig_path = os.path.join(fig_path, dir)
    args.fig_path = fig_path
    #os.mkdir(fig_path)

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

    # With 16 per sub val, 10 WS working, MILC default
    if args.exp == 'FPT':
        gain = [0.45, 0.05, 0.05, 0.15, 0.85]  # FPT
    elif args.exp == 'UFPT':
        gain = [0.05, 0.05, 0.05, 0.05, 0.05]  # UFPT
    else:
        gain = [0.15, 0.25, 0.5, 0.9, 0.35]  # NPT

    sub_per_class = tr_sub[ID]
    current_gain = gain[ID]
    args.gain = current_gain
    sample_x = 100
    sample_y = 20
    subjects = 311
    tc = 140

    samples_per_subject = int(tc / sample_y)
    samples_per_subject = 13
    # samples_per_subject = int((tc - sample_y)+1)
    ntest_samples_perclass = 32
    nval_samples_perclass = 16
    test_start_index = 0
    test_end_index = test_start_index + ntest_samples_perclass
    window_shift = 10

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

    n_good_comp = 53
    finalData2 = np.zeros((subjects, samples_per_subject, n_good_comp, sample_y))

    hf = h5py.File('../FBIRN_AllData.h5', 'r')
    data2 = hf.get('FBIRN_dataset')
    data2 = np.array(data2)
    data2 = data2.reshape(subjects, sample_x, tc)
    data = data2



    if args.fMRI_twoD:

        finalData = data
        finalData = torch.from_numpy(finalData).float()
        finalData = finalData.permute(0, 2, 1)
        finalData = finalData.reshape(finalData.shape[0], finalData.shape[1], finalData.shape[2], 1)
    else:
        finalData = np.zeros((subjects, samples_per_subject, sample_x, sample_y))
        for i in range(subjects):
            for j in range(samples_per_subject):
                finalData[i, j, :, :] = data[i, :, (j * window_shift):(j * window_shift) + sample_y]
        finalData = torch.from_numpy(finalData).float()


    print(finalData.shape)
    filename = 'correct_indices_GSP.csv'
    #print(filename)
    df = pd.read_csv(filename, header=None)
    c_indices = df.values
    c_indices = torch.from_numpy(c_indices).int()
    c_indices = c_indices.view(53)
    c_indices = c_indices - 1
    finalData2 = finalData[:, :, c_indices.long(), :]

    filename = 'index_array_labelled_FBIRN.csv'
    df = pd.read_csv(filename, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).long()
    index_array = index_array.view(subjects)

    filename = 'labels_FBIRN.csv'
    df = pd.read_csv(filename, header=None)
    all_labels = df.values
    all_labels = torch.from_numpy(all_labels).int()
    all_labels = all_labels.view(subjects)
    all_labels = all_labels - 1
    finalData2 = finalData2[index_array, :, :, :]
    all_labels = all_labels[index_array]

    if args.fMRI_twoD:
        finalData2 = finalData2.reshape(finalData2.shape[0], finalData2.shape[1], finalData2.shape[2])
        finalData2 = finalData2.reshape(finalData2.shape[0], finalData2.shape[1], 1, finalData2.shape[2])
    # all_labels = torch.randint(high=2, size=(311,), dtype=torch.int64)
    test_indices = [0, 32, 64, 96, 120]
    HC_index, SZ_index = find_indices_of_each_class(all_labels)
    for test_ID in range(1):

        test_start_index = test_indices[test_ID]
        test_end_index = test_start_index + ntest_samples_perclass
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
        avg_auc = 0.
        for trial in range(ntrials):
                print ('trial = ', trial)

            # writer.add_scalar('trial', trial)
            # current_gain = (trial+1) * 0.05
            # args.gain = current_gain
            # for g_trial in range (ngtrials):
                g_trial=1
                output_text_file = open(output_path, "a+")
                output_text_file.write("Trial = %d gTrial = %d\r\n" % (trial,g_trial))
                output_text_file.close()
                # Get subject_per_class number of random values
                HC_random = torch.randperm(total_HC_index_tr.shape[0])
                SZ_random = torch.randperm(total_SZ_index_tr.shape[0])
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

                #tr_eps = torch.from_numpy(tr_eps).float()
                #val_eps = torch.from_numpy(val_eps).float()
                #test_eps = torch.from_numpy(test_eps).float()

                #tr_labels = tr_labels.to(device)
                #val_labels = val_labels.to(device)
                #test_labels = test_labels.to(device)
                # print('tr1', tr_eps.device, tr_eps.dtype, type(tr_eps), tr_eps.type())
                #tr_eps = tr_eps.to(device)
                #val_eps = val_eps.to(device)
                #test_eps = test_eps.to(device)
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
                # encoder.to(device)
                # lstm_model.to(device)
                # torch.set_num_threads(1)
                complete_model = combinedModel(encoder,lstm_model, gain=current_gain, PT=args.pre_training, exp=args.exp, device=device, oldpath=args.oldpath )
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
                    trainer = LSTMTrainer(complete_model, config, device=device, tr_labels=tr_labels,
                                          val_labels=val_labels, test_labels=test_labels, wandb=wandb, trial=str(trial), gtrial=str(g_trial))

                elif args.method == "global-infonce-stdim":
                    trainer = GlobalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
                elif args.method == "global-local-infonce-stdim":
                    trainer = GlobalLocalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
                elif args.method == "dim":
                    trainer = DIMTrainer(encoder, config, device=device, wandb=wandb)
                else:
                    assert False, "method {} has no trainer".format(args.method)
                xindex = (ntrials * test_ID) + trial
                results[xindex][0], results[xindex][1], results[xindex][2], results[xindex][3] = trainer.pre_train(tr_eps, val_eps, test_eps)
            # auc_arr[g_trial] = trainer.train(tr_eps, val_eps, test_eps)
            # return

        # avg_auc = auc_arr.mean()
        # if avg_auc > best_auc:
        #   best_auc = avg_auc
        #   best_gain = current_gain
    np_results = results.numpy()
    tresult_csv = os.path.join(args.path, 'test_results' + sID + '.csv')
#    np.savetxt(tresult_csv, np_results, delimiter=",")
    #
    # return encoder
    # print ('best gain = ', best_gain)
    # output_text_file = open(output_path, "a+")
    # output_text_file.write("best gain = %f \r\n" % (best_gain))
    # output_text_file.close()
    elapsed = time.time() - start_time
    print('total time = ', elapsed);


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['pretraining-only']
    config = {}
    config.update(vars(args))
    train_encoder(args)
