import time
from collections import deque
from itertools import chain

import numpy as np
import torch
import sys
import os

from src.utils import get_argparser
from src.encoders_ICA import NatureCNN, NatureOneCNN
import pandas as pd
import datetime
from src.encoder_slstm_attn_catalyst import LSTMTrainer
from src.lstm_attn import subjLSTM
from src.All_Architecture_Pre_Training import combinedModel

def train_encoder(args):
    currentDT = datetime.datetime.now()
    d1 = currentDT.strftime("%Y-%m-%d%H:%M:%S")
    dir = 'run-' + d1 + 'milc_1TP'
    wdb='wandb_new'
    path = os.path.join(os.getcwd(), wdb)
    path = os.path.join(path, dir)
    args.path = path
    os.mkdir(path)

    if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        device = torch.device("cuda:" + cudaID)
        # device = torch.device("cuda:" + str(args.cuda_id))
    else:
        device = torch.device("cpu")




    # For ICA TC Data
    data = np.zeros((823, 100, 1040))

    finalData2 = np.zeros((823, 52, 53, 20))

    for p in range(823):
        filename = '../Data/TC/HCP_ica_br' + str(p + 1) + '.csv'
        if p%20 == 0:
            print(filename)
        df = pd.read_csv(filename, header=None)


        d = df.values
        data[p, :, :] = d[:, 0:1040]



    if args.fMRI_twoD:
        finalData = data
        finalData = torch.from_numpy(finalData).float()
        finalData = finalData.reshape(finalData.shape[0], finalData.shape[1], finalData.shape[2], 1)
        finalData = finalData.permute(0,2,1,3)
    else:
        finalData = np.zeros((823, 52, 100, 20))
        for i in range(823):
            for j in range(52):
                finalData[i, j, :, :] = data[i, :, j * 20:j * 20 + 20]
        finalData = torch.from_numpy(finalData).float()

    print(finalData.shape)
    filename = '../IndicesAndLabels/index_array.csv'
    df = pd.read_csv(filename, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).int()
    index_array = index_array.view(823)

    filename = '../IndicesAndLabels/correct_indices_HCP.csv'
    print(filename)
    df = pd.read_csv(filename, header=None)
    c_indices = df.values
    c_indices = torch.from_numpy(c_indices).int()
    c_indices = c_indices.view(53)
    c_indices = c_indices - 1
    finalData2 = finalData[:, :, c_indices.long(), :]
    print(c_indices)


    ID = args.script_ID - 1
    ID = ID * 123

    test_indexs = ID
    test_indexe = ID + 123

    tr_index1 = index_array[0:test_indexs]
    tr_index2 = index_array[test_indexe:823]

    tr_index = torch.cat((tr_index1, tr_index2))

    test_index = index_array[test_indexs:test_indexe]

    if (args.fMRI_twoD):
        finalData2 = finalData2.reshape(finalData2.shape[0], finalData2.shape[1], finalData2.shape[2])
        finalData2 = finalData2.reshape(finalData2.shape[0], finalData2.shape[1], 1, finalData2.shape[2])

    # print(tr_index)
    tr_eps = finalData2[tr_index.long(), :, :, :]
    val_eps = finalData2[500:700, :, :, :]
    test_eps = finalData2[test_index.long(), :, :, :]


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


    elif args.encoder_type == "NatureOne":
        encoder = NatureOneCNN(observation_shape[2], args)

    lstm_model = subjLSTM(device, args.feature_size, args.lstm_size, num_layers=args.lstm_layers,
                          freeze_embeddings=True, gain=0.1)

    config = {}
    config.update(vars(args))
    config['obs_space'] = observation_shape  # weird hack
    complete_model = combinedModel(encoder, lstm_model, gain=0.1, device=device, oldpath=args.oldpath)
    if args.method == "sub-enc-lstm":
        trainer = LSTMTrainer(complete_model, lstm_model, config, device=device, wandb="wandb")
    else:
        assert False, "method {} has no trainer".format(args.method)

    # print("valshape", val_eps.shape)
    trainer.pre_train(tr_eps, val_eps, test_eps)

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
