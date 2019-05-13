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
params = {  "learning_rate": 1e-3,
            "lambdas": [1, 1e-3, 1e-3, 1e-5],
            "switching_epochs": [100000],
            "nb_train": 64*10,
            "nb_val": 1000,
            "validate_every": 1,
            "epochs": 10200,
            "loss2_margin": 1,
            "scoring_hidden_size": 64,
            "batch_size": 64,
            'val_batch_size': 1, 
            "epochs_per_checkpoint": 1,
            "optimizer": "adam",
            "embeddings": "gensim",
            "initial_train_mode": "weights",
            "warm_start": True,
            "starting_i_epoch": 175,
            "pretrained_weights": False,
            "device": "cpu",
            'write_every_epoch': True,
            'write_every_batch': True,
            'stage_searching_epochs': 20,
            'stage_fixing_epochs': 1000,
            'structure_decaying_constant': 1,
            'STRUCTURE_HISTORY_DIR': 'structure_history/',
            'STRUCTURE_OPTIMAL_DIR': 'structure_optimal/',
            'STRUCTURE_HISTORY_VAL': 'structure_val/',
            'CHECKPOINT_DIR': 'checkpoints/',
            'TRAIN_LOSS_FILE': 'train_loss.txt',
            'TRAIN_ACC_FILE': 'train_acc.txt',
            'VAL_LOSS_FILE': 'val_loss.txt',
            'VAL_ACC_FILE': 'val_acc.txt',
            'HYPERPARAM_FILE': 'hyperparameters',
            'N_LOSS_TERMS': 4,
        }

# output folder
if len(sys.argv) != 2:
    dirname = 'test_%s'%(time.asctime().replace(':', '_').replace(' ', '_'))
    dirname = 'test_Sun_May_12_16_43_42_2019'
else:
    dirname = sys.argv[1]
    
if not params['warm_start']:
    os.mkdir(dirname)
os.chdir(dirname)

for path in [params['STRUCTURE_HISTORY_DIR'], params['STRUCTURE_OPTIMAL_DIR'], params['CHECKPOINT_DIR']]:
    if not os.path.isdir(path):
        os.makedirs(path)
        
# Assuming we are already in the directory where the output files should be
if params['warm_start'] is True：:
    params['HYPERPARAM_FILE'] = params['HYPERPARAM_FILE'] + str(params['starting_i_epoch'])
pickle.dump(params, open(params['HYPERPARAM_FILE']+'.pkl', 'wb'))
with open(params['HYPERPARAM_FILE']+'.txt', 'w') as f:
    for key in params.keys():
        f.write('%s: %s\n'%(str(key), str(params[key])))
print('[INFO] Saved hyperparameters.')

device = torch.device(params['device'])
gru_model = torch.load('../gru_parameters_new.pt').to(device)
model = RRNNforGRU(HIDDEN_SIZE, VOCAB_SIZE, batch_size=params['batch_size'],
                   scoring_hsize=params['scoring_hidden_size']).to(device)

# Warm-start with saved weights
if params['warm_start']:
    weights = './checkpoints/checkpoint_epoch%d.pt'%params['starting_i_epoch']
    print('[INFO] Warm starting from ' + weights + '.')
    model.load_state_dict(torch.load(weights))

# data loader
print('[INFO] Loading training data into memory.')
train_set = standard_data.EnWik8Clean(subset='train', n_data=params['nb_train'], device=device)
train_dataloader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=False, drop_last=True)
print('[INFO] Beginning training with %d training samples.'%(len(train_set)))

# train
trainer = RRNNTrainer(model, gru_model, train_dataloader, params)
trainer.train(params['epochs'])
print()
print('[INFO] Run complete')

torch.save(model.state_dict(), 'final_weights.pt')
