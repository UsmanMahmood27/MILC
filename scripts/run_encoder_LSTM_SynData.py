'''
Encoder + LSTM
No attention
'''


import sys
import time
from collections import deque
from itertools import chain


import numpy as np
import torch

import datetime
import os

from src.dim_baseline import DIMTrainer
from src.global_infonce_stdim import GlobalInfoNCESpatioTemporalTrainer
from src.global_local_infonce import GlobalLocalInfoNCESpatioTemporalTrainer
from src.spatio_temporal import SpatioTemporalTrainer
from src.utils import get_argparser
from src.encoders_Mahfuz import NatureCNN, ImpalaCNN, NatureOneCNN, NatureOneNewCNN
from src.cpc import CPCTrainer
from src.vae import VAETrainer
from src.no_action_feedforward_predictor import NaFFPredictorTrainer
from src.infonce_spatio_temporal import InfoNCESpatioTemporalTrainer
from src.slstm_attn import LSTMTrainer
from scripts.load_Augmented_Data import load_Augmented_Data
from src.lstm_attn import subjLSTM

import wandb
import pandas as pd
import random
from aari.episodes import get_episodes



def train_classifier(args):
    if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        device = torch.device("cuda:" + cudaID)
        # device = torch.device("cuda:" + str(args.cuda_id))
    else:
        device = torch.device("cpu")
    print('device = ', device)
    print(torch.cuda.is_available())
    # ID = sys.argv[1]

    # JOB = args.script_ID
    JOB = 1
    args.script_ID = JOB

    #finalData = loadData()
    finalData = load_Augmented_Data()


    # Uncomment these lines if you want to use augmented data
    # Subject wise split of training and test data
    val_eps = np.zeros((200, 200, 10, 20))
    test_eps = np.zeros((200, 200, 10, 20))

    val_eps[0:100, :, :, :] = finalData[800:900, :, :, :]
    val_eps[100:200, :, :, :] = finalData[1800:1900, :, :, :]
    test_eps[0:100, :, :, :] = finalData[900:1000, :, :, :]
    test_eps[100:200, :, :, :] = finalData[1900:2000, :, :, :]

    #=====================================================================================
    # Subject wise split of training ad test data
    # val_eps = np.zeros((60, 200, 10, 20))
    # test_eps = np.zeros((100, 200, 10, 20))
    #
    # val_eps[0:30, :, :, :] = finalData[120:150, :, :, :]
    # val_eps[30:60, :, :, :] = finalData[320:350, :, :, :]
    # test_eps[0:50, :, :, :] = finalData[150:200, :, :, :]
    # test_eps[50:100, :, :, :] = finalData[350:400, :, :, :]
    #====================================================================================

    val_eps = torch.from_numpy(val_eps).float()
    test_eps = torch.from_numpy(test_eps).float()
    val_eps.to(device)
    test_eps.to(device)

    param = {1: 'frozen', 2: 'unfrozen', 3: 'full'}
    exp_type = param[JOB]
    print("Experiment Running:", exp_type)

    subjects_per_group = [100] #, 30, 60, 90, 120]

    #arg_parsed = float(JOB)%len(subjects_per_group)

    Trials = 1
    accuracy = np.zeros([len(subjects_per_group), Trials])
    required_epochs = np.zeros([len(subjects_per_group), Trials])
    loss = np.zeros([len(subjects_per_group), Trials])
    auc = np.zeros([len(subjects_per_group), Trials])
    accuracy = np.zeros([len(subjects_per_group), Trials])

    start_time = time.time()

    for i in range(len(subjects_per_group)):
        total_subjects = subjects_per_group[i] * 2

        print('No. of Subjects in total:', total_subjects)

        for j in range(Trials):
            var_indices = random.sample(range(0, 800),subjects_per_group[i])    #For augmented data, use 0 , 800
            svar_indices = random.sample(range(1000, 1800), subjects_per_group[i]) #For augmented data, use 1000 , 1800

            tr_eps = np.zeros((total_subjects, 200, 10, 20))
            tr_eps[0:subjects_per_group[i], :, :, :] = finalData[var_indices, :, :, :]
            tr_eps[subjects_per_group[i]:total_subjects, :, :, :] = finalData[svar_indices, :, :, :]

            tr_eps = torch.from_numpy(tr_eps).float()
            tr_eps.to(device)

            if j == 0:
                print("trainershape", tr_eps.shape)
                print("valshape", val_eps.shape)

            input_channels = 10
            observation_shape = finalData.shape
            print(observation_shape)
            encoder = MyAutoEncoder(observation_shape[2], args)
            myLSTM = MyLSTM(device, embedding_dim=256, hidden_dim=200, num_layers=1, freeze_embeddings=True)

            if exp_type != 'full':
                path = os.path.join(os.getcwd(), 'wandb', 'pretrained_encoder')
                filename = os.path.join(path, 'autoencOld55'+'.pt')
                model_dict = torch.load(filename)
                encoder.load_state_dict(model_dict)

            encoder.to(device)
            torch.set_num_threads(2)

            config = {}
            config.update(vars(args))

            trainer = LSTMTrainer(encoder, myLSTM, config, device=device, exp=exp_type,  wandb=wandb)
            trainer.to(device)
            accuracy[i, j], loss[i,j], auc[i,j], required_epochs[i, j] = trainer.train(tr_eps, val_eps, test_eps, subjects_per_group[i], j)
            print('Last Trial Accuracy:', accuracy[i, j])
            print('Last Trial AUC:', auc[i, j])
            print('Last Trial Loss:', loss[i, j])
            print('Last Trial Req Epochs:', required_epochs[i, j])

    save_metrics(exp_type, accuracy, loss, auc, required_epochs, run_dir)
    elapsed_time = time.time() - start_time
    print('Total Time Elapsed:', elapsed_time)


    return trainer


if __name__ == "__main__":

    parser = get_argparser()
    args = parser.parse_args()
    #wandb.init()
    config = {}
    config.update(vars(args))
    # wandb.config.update(config)
    train_classifier(args)
