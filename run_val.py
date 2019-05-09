#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 20:21:45 2019
@author: Bruce
"""

from trainer import RRNNTrainer, printable, record_history

import os, sys, platform
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json

from GRU import RRNNforGRU
import pickle
import standard_data

if len(sys.argv) != 2:
    raise ValueError
dirname = sys.argv[1]
if not os.path.isdir(dirname):
    raise ValueError("No such directory: %s"%dirname)
os.chdir(dirname)

VOCAB_SIZE = 27
HIDDEN_SIZE = 100
params = pickle.load(open('hyperparameters.pkl', 'rb'))
print('[INFO] Loaded hyperparameters.')

for path in [params['STRUCTURE_HISTORY_VAL']]:
    if not os.path.isdir(path):
        os.makedirs(path)

device = torch.device(params['device'])
gru_model = torch.load('../gru_parameters_new.pt').to(device)
model = RRNNforGRU(HIDDEN_SIZE, VOCAB_SIZE, batch_size=params['val_batch_size'],
                   scoring_hsize=params['scoring_hidden_size']).to(device)


finished_checkpoints = []
try:
    with open(params['CHECKPOINT_DIR']+'finished.txt', 'r') as f:
        for line in f.readlines():
            finished_checkpoints.append(int(line.lstrip('checkpoint_epoch').rstrip('.pt\n')))
except:
    print('[INFO] No finished checkpoints!')

all_checkpoints = []
for filename in os.listdir(params['CHECKPOINT_DIR']):
    if filename.endswith('.pt'):
        all_checkpoints.append(int(filename.lstrip('checkpoint_epoch').rstrip('.pt')))

avaliable_checkpoints = set(all_checkpoints) - set(finished_checkpoints)
if len(avaliable_checkpoints) > 0:
    i_epoch = max(avaliable_checkpoints)
    filename = 'checkpoint_epoch%d.pt'%(i_epoch)
    with open(params['CHECKPOINT_DIR']+'finished.txt', 'a') as f:
        f.write(filename+'\n')
     
    model.load_state_dict(torch.load(params['CHECKPOINT_DIR']+filename))
    print('[INFO] Warm starting from file: ' + filename)
    
    # data loader
    print('[INFO] Loading validation data into memory.')
    validation_set = standard_data.EnWik8Clean(subset='val', n_data=params['nb_val'], device=device)
    val_dataloader = DataLoader(validation_set, batch_size=params['val_batch_size'], shuffle=False, drop_last=True)
    print('[INFO] Beginning validating with %d samples'%(len(validation_set)))
    
    # validate
    trainer = RRNNTrainer(model, gru_model, val_dataloader, params)
    
    i_batch = -1
    
    timer = time.time()
    for X_batch, y_batch in val_dataloader:
        i_batch += 1
        losses, acc, structures = trainer.train_step_stage_searching(X_batch, y_batch, 'val_%d'%i_epoch, i_batch)
        record_history(params['VAL_LOSS_FILE'], i_epoch, i_batch, printable(losses))
        record_history(params['VAL_ACC_FILE'], i_epoch, i_batch, acc)
        if i_batch%10 == 0:
            print(i_batch, time.time()-timer)
    
    print()
    print('[INFO] Run complete')

