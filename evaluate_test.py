# Script for loading a model's checkpoints and calculating the performance on
# the validation set at each checkpoint.

import sys
import os
import pdb
import numpy as np
import pickle
import torch
import standard_data
import GRU
import trainer
from torch.utils.data import DataLoader
from trainer import RRNNTrainer

# HIDDEN_SIZE = 100
VOCAB_SIZE = 10001
# N_VAL = 1000

# _, (X_val, y_val), _ = standard_data.load_standard_data()
# X_val = X_val[:N_VAL]
# y_val = y_val[:N_VAL]

dirname = sys.argv[1]

checkpoints = [sys.argv[2]]
# for file in os.listdir(dirname):
#     if '.pt' in file:
#         checkpoints.append(file)

state_dicts = [torch.load(dirname + '/' + checkpoint, map_location='cpu') for checkpoint in sorted(checkpoints)]

losses = np.zeros(len(state_dicts))
accuracies = np.zeros_like(losses)
params = pickle.load(open(dirname + '/' + 'hyperparameters.pkl', 'rb'))
gru_model = torch.load('gru_parameters_new.pt', map_location='cpu')
params['n_processes'] = 1
params['debug'] = True
device = 'cpu'
# params['nb_val'] = N_VAL
os.chdir(dirname)
for i, state_dict in enumerate(state_dicts):
    print('[INFO] Performing validation on checkpoint', i)
    # model = GRU.RRNNforGRU(HIDDEN_SIZE, VOCAB_SIZE, scoring_hsize=params['scoring_hidden_size'],
    #                        batch_size=params['batch_size'])
    # for name in state_dict:
    #     print(name)
    # train = trainer.RRNNTrainer(model, gru_model, X_train=None, y_train=None,
    #                             X_val=X_val, y_val=y_val, params=params)
    # train.validate()
    # print('Done')


    if params['dataset'] == 'wiki':
        print('[INFO] Loading the enwik8 dataset.')
        train_set = standard_data.EnWik8Clean(subset='train', n_data=params['nb_train'], device=device)
        validation_set = standard_data.EnWik8Clean(subset='val', n_data=params['nb_val'], device=device)
        test_set = standard_data.EnWik8Clean(subset='test', n_data=2000)
        num_classes = 27
    elif params['dataset'] == 'ptb':
        print('[INFO] Loading the Penn Treebank dataset.')
        dirname = os.getcwd()
        os.chdir('..')
        train_set = standard_data.PennTreebank(subset='train', n_data=params['nb_train'], device=device)
        validation_set = standard_data.PennTreebank(subset='val', n_data=params['nb_val'], device=device)
        test_set = standard_data.PennTreebank(subset='test', n_data=2000)
        os.chdir(dirname)
        num_classes = 10001
    elif params['dataset'] == 'sst':
        print('[INFO] Loading the SST dataset.')
        dirname = os.getcwd()
        os.chdir('..')
        train_set = standard_data.SST(subset='train', n_data=params['nb_train'], device=device)
        validation_set = standard_data.SST(subset='val', n_data=params['nb_val'], device=device)
        test_set = standard_data.SST(subset='test', n_data=2000)
        os.chdir(dirname)
        num_classes = 3
    else:
        raise ValueError('Invalid dataset name. The "dataset" field in the JSON parameter file'
                         ' must be "wiki", "ptb", or "sst".')
    train_dataloader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
    val_dataloader = DataLoader(validation_set, batch_size=params['nb_val'], shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = GRU.RRNNforGRU(100, num_classes, batch_size=params['batch_size'],
                       scoring_hsize=params['scoring_hidden_size']).to(device)
    model.load_state_dict(state_dict)
    trainer = RRNNTrainer(model, gru_model, train_dataloader, val_dataloader, params, test_dataloader)
    trainer.evaluate()
