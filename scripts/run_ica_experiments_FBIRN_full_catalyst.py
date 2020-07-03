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
from src.slstm_attn_catalyst import LSTMTrainer
from src.lstm_attn import subjLSTM
import wandb
import pandas as pd
import datetime
from src.All_Architecture import combinedModel
import nibabel as nib
import matplotlib.pyplot as plt
from aari.episodes import get_episodes
import h5py
from catalyst.dl import utils
# from tensorboardX import SummaryWriter


def find_indices_of_each_class(all_labels):
    HC_index = (all_labels == 0).nonzero()
    SZ_index = (all_labels == 1).nonzero()

    return HC_index, SZ_index


def pre_train():
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['pretraining-only']
    config = {}
    config.update(vars(args))
    train_encoder(args)

def train_encoder(args):
    start_time = time.time()
    # just checking
    # do stuff

    # ID = args.script_ID + 3
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

    Name = args.exp + '_FBIRN_' + args.pre_training + 'fMRI_full_2d1TimePoint_32'
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

    # With 16 per sub val, 10 WS working, default
    # if args.exp == 'FPT':
    #     gain = [0.2, 0.7, 0.95, 0.2, 0.7]     #FPT
    # elif args.exp == 'UFPT':
    #     gain = [0.35, 0.85, 0.6, 0.55, 0.25]    #UFPT
    # else:
    #     gain = [0.3, 0.55, 0.8, 0.6, 0.85]  # NPT

    # With 16 per sub val, 10 WS working, attention default
    # if args.exp == 'FPT':
    #     gain = [0.45, 0.05, 0.05, 0.15, 0.2]     #FPT
    # elif args.exp == 'UFPT':
    #     gain = [0.05, 0.05, 0.05, 0.05, 0.05]    #UFPT
    # else:
    #     gain = [0.15, 0.25, 0.5, 0.9, 0.4]  # NPT

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

    # With 16 per sub val, 10 WS working, MILC default
    if args.exp == 'FPT':
        gain = [0.45, 0.05, 0.05, 0.15, 0.85]  # FPT
    elif args.exp == 'UFPT':
        gain = [0.05, 0.05, 0.05, 0.05, 0.05]  # UFPT
    else:
        gain = [0.15, 0.25, 0.5, 0.9, 0.95]  # NPT

    sub_per_class = tr_sub[ID]
    current_gain = gain[ID]
    args.gain = current_gain
    sample_x = 53
    sample_y = 63
    window_size = 10
    subjects = 311
    tc = 140

    samples_per_subject = int(tc / sample_y)
    samples_per_subject = 27
    # samples_per_subject = int((tc - sample_y)+1)
    ntest_samples_perclass = 32
    nval_samples_perclass = 16
    test_start_index = 0
    test_end_index = test_start_index + ntest_samples_perclass
    window_shift = 5

    if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        device = torch.device("cuda:" + cudaID)
        device_encoder = torch.device("cuda:" + "1")
        # device = torch.device("cuda:" + str(args.cuda_id))
    else:
        device = torch.device("cpu")
        device_encoder = torch.device("cpu")
    print('device = ', device)
    print('device encoder = ', device_encoder)
    # return
    # For ICA TC Data
    fix_z=22
#    data = np.zeros((subjects, sample_x, sample_y, 10, tc))

#    for p in range(subjects):
#            # print(p)
#            filename = '../fMRI/FBIRN/SM' + str(p + 1) + '.nii'
#            # filename = '../TimeSeries/HCP_ica_br1.csv'
#            #if p%20 == 0:
#            print(filename)
#            img = nib.load(filename)


            #img_np = np.array(img.dataobj)
#            img_np = img.get_fdata(caching='unchanged')
#            data[p, :, :, :, :] = img_np[:, :, 17:27, :tc]



    with open('../fMRI/FBIRN/fMRI_full_FBIRN.npz', 'rb') as file:
        finalData2 = np.load(file)

    print('Data loaded')
    finalData2 = torch.from_numpy(finalData2).float()
    finalData2 = finalData2.permute(0, 4, 1, 2, 3)
    if args.fMRI_twoD:
        finalData2 = finalData2.permute(0, 1, 4, 2, 3)
    else:
        finalData2 = finalData2.reshape(finalData2.shape[0], finalData2.shape[1], 1, finalData2.shape[2], finalData2.shape[3], finalData2.shape[4])


    print(finalData2.shape)


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
    finalData2 = finalData2[index_array, :, :, :, :]
    all_labels = all_labels[index_array]
    # all_labels = torch.randint(high=2, size=(311,), dtype=torch.int64)
    test_indices = [32, 32, 64, 96, 120]
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

                tr_index = tr_index.view(tr_index.size(0)).long()
                val_index = val_index.view(val_index.size(0)).long()
                test_index = test_index.view(test_index.size(0)).long()

                #tr_eps = finalData2[tr_index.long(), :, :, :]
                #val_eps = finalData2[val_index.long(), :, :, :]
                #test_eps = finalData2[test_index.long(), :, :, :]
                if (args.fMRI_twoD):
                    tr_eps = finalData2[tr_index, :, :, :, :]
                    val_eps = finalData2[val_index, :, :, :, :]
                    test_eps = finalData2[test_index, :, :, :, :]
                    print('Two D', finalData2.shape)
                else:
                    tr_eps = finalData2[tr_index, :, :, :, :, :]
                    val_eps = finalData2[val_index, :, :, :, :, :]
                    test_eps = finalData2[test_index, :, :, :, :, :]


                tr_labels = all_labels[tr_index]
                val_labels = all_labels[val_index]
                test_labels = all_labels[test_index]

                #tr_eps = torch.from_numpy(tr_eps).float()
                #val_eps = torch.from_numpy(val_eps).float()
                #test_eps = torch.from_numpy(test_eps).float()

                #tr_labels = tr_labels.to(device)
                #val_labels = val_labels.to(device)
                #test_labels = test_labels.to(device)
                # print('tr1', tr_eps.device, tr_eps.dtype, type(tr_eps), tr_eps.type())
                # tr_eps = tr_eps.to(device)
                # val_eps = val_eps.to(device)
                # test_eps = test_eps.to(device)
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
                    encoder = NatureCNN(observation_shape[2], args, device=device, device_one=device_encoder, device_two=device_encoder, device_three=device)
                    if args.pre_training == "milc-fMRI":
                        dir = 'PreTrainedEncoders/Milc_full_3dconv_1timepoint_1_3Chnls/encoder.pt'
                        args.oldpath = wpath1 + '/PreTrainedEncoders/Milc_full_3dconv_1timepoint_1_3Chnls'
                    path = os.path.join(wpath1, dir)
                    lstm_model = subjLSTM(device, args.fMRI_feature_size, args.fMRI_lstm_size, num_layers=args.lstm_layers,
                                          freeze_embeddings=True, gain=current_gain)
                    model_dict = torch.load(path, map_location='cuda')  # with good components

                elif args.encoder_type == "Impala":
                    encoder = ImpalaCNN(observation_shape[2], args)

                # print(encoder)
                if args.exp in ['UFPT', 'FPT']:
                    encoder.load_state_dict(model_dict)
                #encoder.to(device_encoder)
                #lstm_model.to(device)
                #torch.set_num_threads(1)
                complete_model = combinedModel(encoder, lstm_model, gain=current_gain, PT=args.pre_training, exp=args.exp, device=device, oldpath=args.oldpath )


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
                                          val_labels=val_labels, test_labels=test_labels, wandb=wandb, trial=str(trial),
                                          gtrial=str(g_trial))

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
    #tresult_csv = os.path.join(args.path, 'test_results' + sID + '.csv')
    #np.savetxt(tresult_csv, np_results, delimiter=",")
    #
    # return encoder
    # print ('best gain = ', best_gain)
    # output_text_file = open(output_path, "a+")
    # output_text_file.write("best gain = %f \r\n" % (best_gain))
    # output_text_file.close()
    #elapsed = time.time() - start_time
    #print('total time = ', elapsed);


if __name__ == "__main__":
    #utils.distributed_cmd_run(pre_train)
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['pretraining-only']
    config = {}
    config.update(vars(args))
    train_encoder(args)



