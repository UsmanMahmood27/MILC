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
from src.encoders_full_catalyst import NatureCNN, ImpalaCNN
from src.cpc import CPCTrainer
from src.vae import VAETrainer
from src.no_action_feedforward_predictor import NaFFPredictorTrainer
from src.infonce_spatio_temporal import InfoNCESpatioTemporalTrainer
from src.fine_tuning import FineTuning
import wandb
import pandas as pd
import datetime
from src.encoder_slstm_attn_full_catalyst import LSTMTrainer
from src.lstm_attn import subjLSTM
from aari.episodes import get_episodes
import nibabel as nib
import gc
from src.All_Architecture_Pre_Training import combinedModel

def train_encoder(args):
    #print(torch.cuda.nccl.is_available(torch.randn(1).cuda()))
    #print(torch.cuda.nccl.version())




    currentDT = datetime.datetime.now()
    d1 = currentDT.strftime("%Y-%m-%d%H:%M:%S")
    dir = 'run-' + d1 + 'milc_fMRI_full_3dconv_1timepoint_1_3Chnls'
    wdb='wandb_new'

    path = os.path.join(os.getcwd(), wdb)
    path = os.path.join(path, dir)
    args.path = path
    os.mkdir(path)
    numberOfSubjects = 857
    x_dim = 53
    y_dim = 63
    z_dim = 52
    fix_z = 22
    n_windows=52
    time_points = 1040



    if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        print('printing cuda')
        print(cudaID)
        print(torch.cuda.device_count())
        device = torch.device("cuda:0")
        device_one = torch.device("cuda:1")
        device_two = torch.device("cuda:2")
        device_three = torch.device("cuda:3")
        print('device = ', device)
        #print('device_encoder= ', device_encoder)
        # device = torch.device("cuda:" + str(args.cuda_id))
    else:
        device = torch.device("cpu")
        device_one = torch.device("cpu")
        device_two = torch.device("cpu")
        device_three = torch.device("cpu")

#    data = np.zeros((numberOfSubjects, x_dim, y_dim, 10, time_points))

#    with open('../fMRI/HCP/checking.npz', 'wb') as filefmri:
#        np.save(filefmri, data)

#    with open('../fMRI/HCP/checking.npz', 'rb') as file:
#        data1 = np.load(file)

    #finalData = np.zeros((857, 53, 63, 10, 1040))
    with open('../fMRI/HCP/fMRI_full_HCP_Comp.npz', 'rb') as file:
        finalData = np.load(file)
    print('data loaded')


    finalData = torch.from_numpy(finalData).float()
    print(finalData.shape)
    finalData = finalData.permute(0, 4, 1, 2, 3)
    if args.fMRI_twoD:
        finalData = finalData.permute(0, 1, 4, 2, 3)
    else:
        finalData = finalData.reshape(finalData.shape[0], finalData.shape[1], 1, finalData.shape[2], finalData.shape[3], finalData.shape[4])

    print(finalData.shape)


    filename = 'index_array_fMRI.csv'
    df = pd.read_csv(filename, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).int()
    index_array = index_array.view(numberOfSubjects)


    # finalData2 = finalData

    # index_array = torch.randperm(823)
    # index_array = torch.randperm(823)
    # tr_index = index_array[0:500]
    # val_index = index_array[0:500]
    # test_index = index_array[0:500]

    ID = args.script_ID - 1
    ID = ID * 157
    # val_indexs = ID-1;
    # val_indexe = ID+200

    test_indexs = ID
    test_indexe = ID + 157

    tr_index1 = index_array[0:test_indexs]
    tr_index2 = index_array[test_indexe:numberOfSubjects]

    tr_index = torch.cat((tr_index1, tr_index2))

    test_index = index_array[test_indexs:test_indexe]
    # print(tr_index)
    tr_index = tr_index.long()
    test_index = test_index.long()

    if (args.fMRI_twoD):
        tr_eps = finalData[tr_index, :, :, :, :]
        val_eps = "finalData[500:700, :, :, :, :, :]"
        test_eps = finalData[test_index, :, :, :, :]
        print('Two D', finalData.shape)
    else:
        tr_eps = finalData[tr_index, :, :, :, :, :]
        val_eps = "finalData[500:700, :, :, :, :, :]"
        test_eps = finalData[test_index, :, :, :, :, :]

    # tr_eps = torch.from_numpy(tr_eps).float()
    # val_eps = torch.from_numpy(val_eps).float()
    # test_eps = torch.from_numpy(test_eps).float()
    # input_data = torch.rand(20, 10, 1, 160, 210)
    # tr_eps = torch.rand(20, 10, 1, 160, 210)
    # val_eps = torch.rand(20, 10, 1, 160, 210)
    # observation_shape = tr_eps[0][0].shape

    #tr_eps.to(device)
    #test_eps.to(device)

    print("index_arrayshape", index_array.shape)
    print("trainershape", tr_eps.shape)
    #print("valshape", val_eps.shape)
    print("testshape", test_eps.shape)
    initial_n_channels = 1
    print('ID = ', args.script_ID)

    observation_shape = finalData.shape
    if args.encoder_type == "Nature":
        encoder = NatureCNN(observation_shape[2], args, device=device, device_one=device_one, device_two=device_two, device_three=device_three)
        # encoder = NatureCNN(input_channels, args)

    elif args.encoder_type == "Impala":
        encoder = ImpalaCNN(observation_shape[2], args)


    lstm_model = subjLSTM(device, args.fMRI_feature_size, args.fMRI_lstm_size, num_layers=args.lstm_layers,
                          freeze_embeddings=True, gain=0.1)



    #if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #    encoder = torch.nn.DataParallel(encoder)
    #    lstm_model = torch.nn.DataParallel(lstm_model)

    #encoder.to(device_encoder)
    #lstm_model.to(device)
    complete_model = combinedModel(encoder, lstm_model, gain=0.1, PT=args.pre_training, exp=args.exp,
                                   device=device, oldpath=args.oldpath)
    #.set_num_threads(1)

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
    elif args.method == "sub-enc-lstm":
        #trainer = LSTMTrainer(encoder, lstm_model, config, device=device, device_one=device_one, wandb=wandb)
        trainer = LSTMTrainer(complete_model, config, device=device, wandb=wandb)
    elif args.method == "fine-tuning":
        trainer = FineTuning(encoder, config, device=device, wandb=wandb)
    elif args.method == "global-infonce-stdim":
        trainer = GlobalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "global-local-infonce-stdim":
        trainer = GlobalLocalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "dim":
        trainer = DIMTrainer(encoder, config, device=device, wandb=wandb)
    else:
        assert False, "method {} has no trainer".format(args.method)

    # print("valshape", val_eps.shape)
    print('training')
    trainer.train(tr_eps, test_eps)

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
