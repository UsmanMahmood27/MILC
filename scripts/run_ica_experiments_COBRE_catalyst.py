import time
from collections import deque
from itertools import chain

import numpy as np
import torch
import sys
import os

from src.utils import get_argparser
from src.encoders_ICA import NatureCNN, NatureOneCNN
from src.slstm_attn_catalyst import LSTMTrainer
from src.lstm_attn import subjLSTM
from src.All_Architecture import combinedModel
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import h5py


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
    # ID = 2
    print('ID = ' + str(ID))
    print('exp = ' + args.exp)
    print('pretraining = ' + args.pre_training)
    sID = str(ID)
    currentDT = datetime.datetime.now()
    d1 = currentDT.strftime("%Y-%m-%d%H:%M:%S")
    d2 = str(JobID) + '_' + str(ID)

    Name = args.exp + '_FBIRN_' + args.pre_training
    dir = 'run-' + d1 + d2 + Name
    dir = dir + '-' + str(ID)
    wdb = 'wandb_new'
    output_path = "Output"
    opath = os.path.join(os.getcwd(), output_path)
    args.path = opath

    wdb1 = 'wandb_new'
    wpath1 = os.path.join(os.getcwd(), wdb1)

    tfilename = str(JobID) + 'outputFILE' + Name + str(ID)

    output_path = os.path.join(os.getcwd(), 'Output')
    output_path = os.path.join(output_path, tfilename)

    ntrials = 1
    ngtrials = 10
    tr_sub = [15, 25, 40]
    sub_per_class = tr_sub[ID]
    sample_x = 100
    sample_y = 20
    subjects = 157
    tc = 140
    # samples_per_subject = int(tc / sample_y)
    samples_per_subject = 7
    # samples_per_subject = int((tc - sample_y)+1)
    # ntest_samples = 32
    window_shift = 20
    n_val_HC = 14
    n_val_SZ = 16
    n_test_HC = 14
    n_test_SZ = 16
    if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        device = torch.device("cuda:" + cudaID)
        #device = torch.device("cuda:" + str(1))
    else:
        device = torch.device("cpu")
    print('device = ', device)
    # Milc default
    if args.exp == 'FPT':
        gain = [0.1, 0.05, 0.05]  # FPT
    elif args.exp == 'UFPT':
        gain = [0.05, 0.45, 0.65]  # UFPT
    else:
        gain = [0.25, 0.35, 0.65]  # NPT

    current_gain = gain[ID]
    args.gain = current_gain

    hf = h5py.File('../Data/COBRE_AllData.h5', 'r')
    data = hf.get('COBRE_dataset')
    data = np.array(data)
    data = data.reshape(subjects, sample_x, tc)

    # Get Training indices for cobre and convert them to tensor. this is to have same training samples everytime.
    hf_hc = h5py.File('../IndicesAndLabels/COBRE_HC_TrainingIndex.h5', 'r')
    HC_TrainingIndex = hf_hc.get('HC_TrainingIndex')
    HC_TrainingIndex = np.array(HC_TrainingIndex)
    HC_TrainingIndex = torch.from_numpy(HC_TrainingIndex)

    hf_sz = h5py.File('../IndicesAndLabels/COBRE_SZ_TrainingIndex.h5', 'r')
    SZ_TrainingIndex = hf_sz.get('SZ_TrainingIndex')
    SZ_TrainingIndex = np.array(SZ_TrainingIndex)
    SZ_TrainingIndex = torch.from_numpy(SZ_TrainingIndex)

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
    filename = '../IndicesAndLabels/correct_indices_GSP.csv'
    # print(filename)
    df = pd.read_csv(filename, header=None)
    c_indices = df.values
    c_indices = torch.from_numpy(c_indices).int()
    c_indices = c_indices.view(53)
    c_indices = c_indices - 1
    finalData2 = finalData[:, :, c_indices.long(), :]

    filename = '../IndicesAndLabels/index_array_labelled_COBRE.csv'
    df = pd.read_csv(filename, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).long()
    index_array = index_array.view(subjects)

    filename = '../IndicesAndLabels/labels_COBRE.csv'
    df = pd.read_csv(filename, header=None)
    df = pd.read_csv(filename, header=None)
    all_labels = df.values
    all_labels = torch.from_numpy(all_labels).int()
    all_labels = all_labels.view(subjects)
    all_labels = all_labels - 1
    finalData2 = finalData2[index_array, :, :, :]
    all_labels = all_labels[index_array]

    HC_index, SZ_index = find_indices_of_each_class(all_labels)

    total_HC_index_tr = HC_index[:len(HC_index) - (n_val_HC + n_test_HC)]
    total_SZ_index_tr = SZ_index[:len(SZ_index) - (n_val_SZ + n_test_SZ)]


    HC_index_val = HC_index[len(HC_index) - (n_val_HC + n_test_HC):len(HC_index) - n_test_HC]
    SZ_index_val = SZ_index[len(HC_index) - (n_val_SZ + n_test_SZ):len(HC_index) - n_test_SZ]

    HC_index_test = HC_index[len(HC_index) - (n_test_HC):]
    SZ_index_test = SZ_index[len(SZ_index) - (n_test_SZ):]
    results = torch.zeros(ntrials, 4)
    auc = torch.zeros(ntrials, 1)
    sss=0
    #HCrand=[]
    #SZrand=[]
    for trial in range(ntrials):
        print('trial = ', trial)
        output_text_file = open(output_path, "a+")
        output_text_file.write("Trial = %d gTrial = %d\r\n" % (trial, 1))
        output_text_file.close()
        # Get subject_per_class number of random values


        HC_random = HC_TrainingIndex[trial]
        SZ_random = SZ_TrainingIndex[trial]
        HC_random = HC_random[:sub_per_class]
        SZ_random = SZ_random[:sub_per_class]
        #

        # Choose the subject_per_class indices from HC_index_val and SZ_index_val using random numbers

        HC_index_tr = total_HC_index_tr[HC_random]
        SZ_index_tr = total_SZ_index_tr[SZ_random]



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


        observation_shape = finalData2.shape
        if args.encoder_type == "Nature":
            encoder = NatureCNN(observation_shape[2], args)

        elif args.encoder_type == "NatureOne":
            dir = ""
            if args.pre_training == "basic":
                dir = 'PreTrainedEncoders/Basic_Encoder'
            elif args.pre_training == "milc":
                args.oldpath = wpath1 + '/PreTrainedEncoders/Milc'

            encoder = NatureOneCNN(observation_shape[2], args)
            lstm_model = subjLSTM(device, args.feature_size, args.lstm_size, num_layers=args.lstm_layers,
                                  freeze_embeddings=True, gain=current_gain)

        complete_model = combinedModel(encoder, lstm_model, gain=current_gain, PT=args.pre_training,
                                       exp=args.exp, device=device, oldpath=args.oldpath,
                                       complete_arc=args.complete_arc)
        config = {}
        config.update(vars(args))
        config['obs_space'] = observation_shape

        if args.method == "sub-lstm":
            trainer = LSTMTrainer(complete_model, config, device=device, tr_labels=tr_labels,
                                  val_labels=val_labels, test_labels=test_labels, wandb="wandb", trial=str(trial))
        elif args.method == "sub-enc-lstm":
            print("Change method to sub-lstm")
        else:
            assert False, "method {} has no trainer".format(args.method)

        results[trial][0], results[trial][1], results[trial][2], results[trial][3] = trainer.pre_train(tr_eps,
                                                                                                               val_eps,
                                                                                                               test_eps)

    np_results = results.numpy()
    tresult_csv = os.path.join(args.path, 'test_results' + sID + '.csv')
    np.savetxt(tresult_csv, np_results, delimiter=",")
    elapsed = time.time() - start_time
    print('total time = ', elapsed);


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['pretraining-only']
    config = {}
    config.update(vars(args))
    train_encoder(args)
