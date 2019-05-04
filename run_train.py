#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 20:21:45 2019
@author: Bruce
"""

from trainer import RRNNTrainer

import os, sys, platform
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json

from GRU import RRNNforGRU
import pickle
import standard_data


VOCAB_SIZE = 27
HIDDEN_SIZE = 100
params = {  "learning_rate": 1e-4,
            "lambdas": [1, 1e-3, 1e-3, 1e-5],
            "nb_train": 64,
            "nb_val": 1,
            "validate_every": 1,
            "epochs": 10200,
            "loss2_margin": 1,
            "scoring_hidden_size": 64,
            "batch_size": 64,
            "epochs_per_checkpoint": 1,
            "optimizer": "adam",
            "embeddings": "gensim",
            "initial_train_mode": "weights",
            "warm_start": False,
            "pretrained_weights": False,
            "device": "cpu",
            'write_every_epoch': True,
            'write_every_batch': True,
            'stage_searching_epochs': 20,
            'stage_fixing_epochs': 1000,
            'structure_decaying_constant': 1,
            'STRUCTURE_HISTORY_DIR': 'structure_history/',
            'STRUCTURE_OPTIMAL_DIR': 'structure_optimal/',
            'CHECKPOINT_DIR': 'checkpoints/',
            'TRAIN_LOSS_FILE': 'train_loss.txt',
            'TRAIN_ACC_FILE': 'train_acc.txt',
            'VAL_LOSS_FILE': 'val_loss.txt',
            'VAL_ACC_FILE': 'val_acc.txt',
            'HYPERPARAM_FILE': 'hyperparameters.pkl',
            'N_LOSS_TERMS': 4,
        }

# output folder
if len(sys.argv) != 2:
    dirname = 'test %s'%(time.asctime().replace(':', '_'))
else:
    dirname = sys.argv[1]
if not params['warm_start']:
    os.mkdir(dirname)
os.chdir(dirname)
for path in [params['STRUCTURE_HISTORY_DIR'], params['STRUCTURE_OPTIMAL_DIR'], params['CHECKPOINT_DIR']]:
    if not os.path.isdir(path):
        os.makedirs(path)
        
# Assuming we are already in the directory where the output files should be
pickle.dump(params, open(params['HYPERPARAM_FILE'], 'wb'))
with open('hyperparameters.txt', 'w') as f:
    f.write(str(params)
print('[INFO] Saved hyperparameters.')

device = torch.device(params['device'])
gru_model = torch.load('../gru_parameters.pkl').to(device)
model = RRNNforGRU(HIDDEN_SIZE, VOCAB_SIZE, batch_size=params['batch_size'],
                   scoring_hsize=params['scoring_hidden_size']).to(device)

# Warm-start with saved weights
if params['warm_start']:
    weights = params['weights_file']
    print('[INFO] Warm starting from ' + weights + '.')
    model.load_state_dict(torch.load(weights))

# data loader
print('[INFO] Loading training data into memory.')
train_set = standard_data.EnWik8Clean(subset='train', n_data=params['nb_train'], device=device)
validation_set = standard_data.EnWik8Clean(subset='val', n_data=params['nb_val'], device=device)
train_dataloader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=False, drop_last=True)
val_dataloader = DataLoader(validation_set, batch_size=params['nb_val'], shuffle=False)
print('[INFO] Beginning training with %d training samples and %d '
      'validation samples.' % (len(train_set), len(validation_set)))

# train
trainer = RRNNTrainer(model, gru_model, train_dataloader, val_dataloader, params)
trainer.train(params['epochs'])
print()
print('[INFO] Run complete')

torch.save(model.state_dict(), 'final_weights.pt')
