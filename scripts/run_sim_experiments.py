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
from src.encoders_Mahfuz import NatureCNN, ImpalaCNN, NatureOneCNN
from src.cpc import CPCTrainer
from src.vae import VAETrainer
from src.no_action_feedforward_predictor import NaFFPredictorTrainer
from src.infonce_spatio_temporal import InfoNCESpatioTemporalTrainer
from src.fine_tuning import FineTuning
from src.slstm import LSTMTrainer
from src.lstm import subjLSTM
from src.autoencoder import MyAutoEncoder
import wandb
import pandas as pd
import datetime
from scripts.load_Augmented_Data import load_Augmented_Data
import random
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

    ID = 4
    print('ID = ' + str(ID))
    print('exp = ' + args.exp)
    print('pretraining = ' + args.pre_training)
    sID = str(ID)
    currentDT = datetime.datetime.now()
    d1 = currentDT.strftime("%Y-%m-%d%H:%M:%S")
    d2 = str(JobID) + '_' + str(ID)

    Name = args.exp + '_SIM_var_svar_STDIM_Second' + args.pre_training
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
    per_to_keep = 0.8

    # p = 'UF'
    # dir = 'run-2019-09-1223:36:31' + '-' + str(ID)+'UF'
    # p_path = os.path.join(os.getcwd(), p)
    # p_path = os.path.join(p_path, dir)
    # args.p_path = p_path
    # os.mkdir(fig_path)


    ntrials = 10
    tr_sub = [15, 30, 60, 90, 120]
    if args.exp == 'FPT':
        gain = [0.55, 0.95, 0.25, 0.5, 0.5, 0.3]     #FPT
    elif args.exp == 'UFPT':
        gain = [0.4, 0.2, 0.25, 0.2, 0.35, 0.35]    #UFPT
    else:
        gain = [0.4, 0.5, 0.75, 0.75, 0.45, 0.8]     #NPT

    tfilename = str(JobID) + 'outputFILE' + Name + str(ID)

    output_path = os.path.join(os.getcwd(), 'Output')
    output_path = os.path.join(output_path, tfilename)

    sub_per_class = tr_sub[ID]
    current_gain = gain[ID]
    sample_x = 10
    sample_y = 20
    subjects = 2000
    subjects_for_train = sub_per_class * 2
    subjects_for_test = 200
    subjects_for_val = 200
    tc = 20000
    samples_per_subject = int(tc / sample_y)
    samples_per_subject = 200
    # samples_per_subject = int((tc - sample_y)+1)
    ntrain_samples = 200
    ntest_samples = 200
    window_shift = 20
    if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        device = torch.device("cuda:" + cudaID)
        # device = torch.device("cuda:" + str(args.cuda_id))
    else:
        device = torch.device("cpu")
    print('device = ', device)
    ndata=1
    create_data = 0
    if ndata == 1:

        windowsize = 20
        windows = int((per_to_keep * 4000) / windowsize)
        data = np.zeros((2000, 10, 4000))
        finalData = np.zeros((2000, windows, 10, windowsize))
        sample_x = 53
        finalData2 = np.zeros((2000, windows, 10, windowsize))
        if create_data == 1:

            # For ICA TC Data

            if not os.path.exists('../SIM_Usman.npz'):

                for p in range(400):
                    filename = '../NewMoreSamplesVARandSVARTimeSeries/TSDataCSV' + str(p) + '.csv'
                    if (p % 100) == 0:
                        print(filename)
                    df = pd.read_csv(filename)
                    df = df.to_numpy()
                    data[p * 5:p * 5 + 5, :, :] = df.reshape(5, 10,4000)  # Five slices per TS save them as 5 TS samples of shorter length

                with open('../SIM_Usman.npz', 'wb') as filesim:
                    np.save(filesim, data)
            else:
                with open('../SIM_Usman.npz', 'rb') as file:
                    data = np.load(file)
                    print('data loaded successfully...')
            var = data[0:1000,:,:]
            var = torch.from_numpy(var)
            svar = []
            val = int(per_to_keep * 4000)
            for i in range(1000):
                rand = random.sample(range(4000), val)
                rand.sort()
                svar.append(var[i,:,rand])
            var = var[:,:,:val]
            svar = torch.stack(svar)
            data = torch.cat((var,svar))
            with open('../SIM_Usman_0point8Var.npz', 'wb') as filesimvar:
                np.save(filesimvar, data)
        else:
            with open('../SIM_Usman_0point8Var.npz', 'rb') as file:
                data = np.load(file)



        for i in range(400):
           for j in range(windows):
                finalData[i * 5:i * 5 + 5, j, :, :] = data[i * 5:i * 5 + 5, :, j * windowsize:j * windowsize + windowsize]


    else:
        finalData = load_Augmented_Data()

    print(finalData.shape)

    hf_hc = h5py.File('../SIM_HC_TrainingIndex.h5', 'r')
    HC_TrainingIndex = hf_hc.get('HC_TrainingIndex')
    HC_TrainingIndex = np.array(HC_TrainingIndex)
    HC_TrainingIndex = torch.from_numpy(HC_TrainingIndex)

    hf_sz = h5py.File('../SIM_SZ_TrainingIndex.h5', 'r')
    SZ_TrainingIndex = hf_sz.get('SZ_TrainingIndex')
    SZ_TrainingIndex = np.array(SZ_TrainingIndex)
    SZ_TrainingIndex = torch.from_numpy(SZ_TrainingIndex)




    vl = torch.zeros(1000)
    svl= torch.ones(1000)
    all_labels = torch.cat((vl,svl))

    # all_labels = torch.randint(high=2, size=(311,), dtype=torch.int64)

    HC_index, SZ_index = find_indices_of_each_class(all_labels)
    total_HC_index_tr = HC_index[:len(HC_index) - (100 + 100)]
    total_SZ_index_tr = SZ_index[:len(SZ_index) - (100 + 100)]

    HC_index_val = HC_index[len(HC_index) - (100 + 100):len(HC_index) - 100]
    SZ_index_val = SZ_index[len(HC_index) - (100 + 100):len(HC_index) - 100]

    HC_index_test = HC_index[len(HC_index) - (100):]
    SZ_index_test = SZ_index[len(SZ_index) - (100):]
    results = torch.zeros(ntrials, 4)
    auc = torch.zeros(ntrials, 1)
    # HCrand=[]
    # SZrand=[]
    for trial in range(ntrials):
    # Get subject_per_class number of random values
        print('trial = ', trial);
        output_text_file = open(output_path, "a+")
        output_text_file.write("Trial = %d\r\n" % (trial))
        output_text_file.close()

        # HCrand.append(torch.randperm(total_HC_index_tr.shape[0]))
        # SZrand.append(torch.randperm(total_SZ_index_tr.shape[0]))

        HC_random = HC_TrainingIndex[trial]
        SZ_random = SZ_TrainingIndex[trial]
        HC_random = HC_random[:sub_per_class]
        SZ_random = SZ_random[:sub_per_class]
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

        tr_eps = finalData[tr_index, :, :, :]
        val_eps = finalData[val_index, :, :, :]
        test_eps = finalData[test_index, :, :, :]

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
        observation_shape = finalData.shape
        if args.encoder_type == "Nature":
            encoder = NatureCNN(observation_shape[2], args)
        # encoder = NatureCNN(input_channels, args)

        elif args.encoder_type == "Impala":
            encoder = ImpalaCNN(observation_shape[2], args)


        elif args.encoder_type == "NatureOne":
            # encoder = NatureOneCNN(observation_shape[2], args)
            dir = ""
            if args.pre_training == "basic":
                dir = 'PreTrainedEncoders/SIM/STDIM/encoder.pt'
            elif args.pre_training == "AE":
                dir = 'PreTrainedEncoders/SIM/AE/autoencOldFinal.pt'
            elif args.pre_training == "milc":
                dir = 'PreTrainedEncoders/SIM/Milc/encoder.pt'
                args.oldpath = wpath1 + '/PreTrainedEncoders/SIM/Milc'
            elif args.pre_training == "two-loss-milc":
                dir = 'PreTrainedEncoders/SIM/MilcWithTwoLosses/encoder.pt'
                args.oldpath = wpath1 + '/PreTrainedEncoders/SIM/MilcWithTwoLosses'
            elif args.pre_training == "variable-attention":
                dir = 'PreTrainedEncoders/SIM/VariableAttention/encoder.pt'
                args.oldpath = wpath1 + '/PreTrainedEncoders/SIM/VariableAttention'
            path = os.path.join(wpath1, dir)
            # dir = 'PreTrainedEncoders/VariableAttention/encoder.pt'
            # args.oldpath = wpath1 + '/run-2020-01-0401:25:58'
            # path = os.path.join(wpath1, dir)
            encoder = NatureOneCNN(observation_shape[2], args)
            # encoder = MyAutoEncoder(observation_shape[2], args)
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
            print('in ufpt and fpt')
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
            trainer = FineTuning(encoder, config, device=device, tr_labels=tr_labels, test_labels=test_labels, wandb=wandb)
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


    # print("valshape", val_eps.shape)
        results[trial][0], results[trial][1], results[trial][2], results[trial][3]= trainer.train(tr_eps, val_eps, test_eps)
        # auc[trial][0]=trainer.validate(val_eps)
    np_results = results.numpy()
    tresult_csv = os.path.join(args.path, 'test_results'+ sID +'.csv')
    np.savetxt(tresult_csv, np_results, delimiter=",")

    # np_auc = auc.numpy()
    # np.savetxt(tresult_csv, np_auc, delimiter=",")

    # return encoder

    # hfn = h5py.File('../SIM_HC_TrainingIndex.h5', 'w')
    # hfn.create_dataset('HC_TrainingIndex', data=torch.stack(HCrand))
    # hfn.close()
    #
    # hfq = h5py.File('../SIM_SZ_TrainingIndex.h5', 'w')
    # hfq.create_dataset('SZ_TrainingIndex', data=torch.stack(SZrand))
    # hfq.close()


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
