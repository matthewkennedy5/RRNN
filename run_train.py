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


# Performs a training run using the given hyperparameters. Saves out data and model checkpoints
# into the current directory.
def run(params):
    # Assuming we are already in the directory where the output files should be
    pickle.dump(params, open(params['HYPERPARAM_FILE'], 'wb'))
    print('[INFO] Saved hyperparameters.')

    device = torch.device(params['device'])
    gru_model = torch.load('../gru_parameters.pkl').to(device)

    # Extract GRU pre-trained weights
    W_ir, W_iz, W_in = gru_model.weight_ih_l0.chunk(3)
    W_hr, W_hz, W_hn = gru_model.weight_hh_l0.chunk(3)
    b_ir, b_iz, b_in = gru_model.bias_ih_l0.chunk(3)
    b_hr, b_hz, b_hn = gru_model.bias_hh_l0.chunk(3)

    L1 = W_ir
    R1 = W_hr
    b1 = b_ir + b_hr
    L2 = W_iz
    R2 = W_hz
    b2 = b_iz + b_hz
    L3 = W_in
    R3 = W_hn
    b3 = b_in #+ r*b_hn

    model = RRNNforGRU(HIDDEN_SIZE, VOCAB_SIZE, batch_size=params['batch_size'],
                       scoring_hsize=params['scoring_hidden_size']).to(device)

    # Warm-start with pretrained GRU weights
    if params['pretrained_weights']:
        print('[INFO] Loading pre-trained GRU weights.')
        model.cell.L_list[1] = nn.Parameter(L1)
        model.cell.L_list[2] = nn.Parameter(L2)
        model.cell.L_list[3] = nn.Parameter(L3)
        model.cell.R_list[1] = nn.Parameter(R1)
        model.cell.R_list[2] = nn.Parameter(R2)
        model.cell.R_list[3] = nn.Parameter(R3)
        model.cell.b_list[1] = nn.Parameter(b1)
        model.cell.b_list[2] = nn.Parameter(b2)
        model.cell.b_list[3] = nn.Parameter(b3)

    if params['warm_start']:
        weights = params['weights_file']
        print('[INFO] Warm starting from ' + weights + '.')
        model.load_state_dict(torch.load(weights))

    print('[INFO] Loading training data into memory.')
    train_set = standard_data.EnWik8Clean(subset='train', n_data=params['nb_train'], device=device)
    validation_set = standard_data.EnWik8Clean(subset='val', n_data=params['nb_val'], device=device)
    train_dataloader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=False, drop_last=True)
    val_dataloader = DataLoader(validation_set, batch_size=params['nb_val'], shuffle=False)
    print('[INFO] Beginning training with %d training samples and %d '
          'validation samples.' % (len(train_set), len(validation_set)))

    trainer = RRNNTrainer(model, gru_model, train_dataloader, val_dataloader, params)
    trainer.train(params['epochs'])
    print()
    print('[INFO] Run complete')

    torch.save(model.state_dict(), 'final_weights.pt')
    return trainer

if __name__ == '__main__':

    if platform.system() in ['Windows', 'Darwin']:
        dirname = 'test %s'%(time.asctime().replace(':', '_'))
        params = {
                    "learning_rate": 1e-4,
                    "lambdas": [1, 1e-3, 1e-3, 1e-5],
                    "nb_train": 128,
                    "nb_val": 10,
                    "validate_every": 1,
                    "epochs": 20,
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
                    'stage_searching_epochs': 3,
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
        if not params['warm_start']:
            os.mkdir(dirname)
        os.chdir(dirname)
        
        for path in [params['STRUCTURE_HISTORY_DIR'], params['STRUCTURE_OPTIMAL_DIR'], params['CHECKPOINT_DIR']]:
            if not os.path.isdir(path):
                os.makedirs(path)
            
        trainer = run(params)
        model = trainer.model

    else: # elif platform.system() == '' # on server
        if len(sys.argv) != 3:
            raise Exception('Usage: python trainer.py <output_dir> <JSON parameter file>')
        dirname = sys.argv[1]
        param_file = sys.argv[2]
        with open(param_file, 'r') as f:
            params = json.load(f)

        if not params['warm_start']:
            os.mkdir(dirname)
        os.chdir(dirname)
        
        for path in [params['STRUCTURE_HISTORY_DIR'], params['STRUCTURE_OPTIMAL_DIR'], params['CHECKPOINT_DIR']]:
            if not os.path.isdir(path):
                os.makedirs(path)
        
        trainer = run(params)
        model = trainer.model
