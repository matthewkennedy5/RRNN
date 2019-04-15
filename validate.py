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

HIDDEN_SIZE = 100
VOCAB_SIZE = 27
N_VAL = 1000

_, (X_val, y_val), _ = standard_data.load_standard_data()
X_val = X_val[:N_VAL]
y_val = y_val[:N_VAL]

dirname = sys.argv[1]

checkpoints = []
for file in os.listdir(dirname):
    if '.pt' in file:
        checkpoints.append(file)

state_dicts = [torch.load(dirname + '/' + checkpoint) for checkpoint in sorted(checkpoints)]

losses = np.zeros(len(state_dicts))
accuracies = np.zeros_like(losses)
params = pickle.load(open(dirname + '/' + 'hyperparameters.pkl', 'rb'))
gru_model = torch.load('gru_parameters.pkl')
params['n_processes'] = 1
params['debug'] = True
params['nb_val'] = N_VAL
os.chdir(dirname)
for i, state_dict in enumerate(state_dicts):
    print('[INFO] Performing validation on checkpoint', i)
    model = GRU.RRNNforGRU(HIDDEN_SIZE, VOCAB_SIZE, params['scoring_hidden_size'])
    model.load_state_dict(state_dict)
    train = trainer.RRNNTrainer(model, gru_model, X_train=None, y_train=None,
                                X_val=X_val, y_val=y_val, params=params)
    train.validate()
    print('Done')
