# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:50:17 2019

@author: Bruce
"""

import os
import sys
import time
import datetime
import random
import torch
from torch import nn
import torch.multiprocessing as mp
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# import tree_methods_parallel as tree_methods
import tree_methods
from GRU import RRNNforGRU
import pickle
import pdb
import standard_data

VOCAB_SIZE = 27
HIDDEN_SIZE = 100
n_epoch = 100
time_steps = 20


TRAIN_LOSS_FILE = 'loss.txt'
TRAIN_ACC_FILE = 'train_acc.txt'
VAL_LOSS_FILE = 'val_loss.txt'
VAL_ACC_FILE = 'val_acc.txt'
HYPERPARAM_FILE = 'hyperparameters.pkl'
RUNTIME_FILE = 'runtime.pkl'

params = {
            "learning_rate": 1e-4,
            "multiplier": 1,
            "lambdas": [1, 0, 1e-8, 0.003],
            "nb_train": 10000,
            "nb_val": 10,
            "validate_every": 1,
            "epochs": 2,
            "loss2_margin": 1,
            "scoring_hidden_size": 64,
            "batch_size": 64,
            "epochs_per_checkpoint": 1,
            "pickle_every": 1,
            "optimizer": "adam",
            "embeddings": "gensim",
            "max_grad": 1,
            "initial_train_mode": "weights",
            "alternate_every": 1,
            "warm_start": False,
            "weights_file": "epoch_0.pt",
            "pretrained_weights": False,
            "device": "cpu"
        }

batch_size = params['batch_size']
device = params['device']
dirname = 'test%s'%(time.asctime().replace(':', '_'))
if not params['warm_start']:
    os.mkdir(dirname)
os.chdir(dirname)

gru_model = torch.load('../gru_parameters.pkl')
model = RRNNforGRU(HIDDEN_SIZE, VOCAB_SIZE, batch_size, params['scoring_hidden_size'])
#filename = os.path.join('..', params['data_file'])

train_set = standard_data.EnWik8Clean(subset='train', n_data=params['nb_train'], device=device)
train_dataloader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
lamb1, lamb2, lamb3, lamb4 = params['lambdas']
loss = torch.nn.CrossEntropyLoss()

if params['optimizer'] == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
elif params['optimizer'] == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'])

if params['device'] != 'cpu':
    gru_model = gru_model.cuda()
    model = model.cuda()

model.train()
gru_model.train()

timer=time.time()

for i_epoch in range(n_epoch):
    i_batch = -1
    for X, y in train_dataloader:
        i_batch += 1
        optimizer.zero_grad()
        
        # forward pass
        pred_chars_batch, h_batch, pred_tree_list, structures_list, margins_batch = model(X)
        
        # forward pass of traditional GRU
        gru_h_list = gru_model(X)[0]
        gru_h_list = torch.cat([torch.zeros([batch_size, 1, HIDDEN_SIZE], device=X.device), gru_h_list], dim=1)
        target_tree_list = []
        for t in range(X.shape[1]):
            gru_x = X[:, t, :].reshape(batch_size, 1, HIDDEN_SIZE)
            gru_h = gru_h_list[:, t, :].reshape(batch_size, 1, HIDDEN_SIZE)
            target_tree = tree_methods.GRUtree_pytorch(gru_x, gru_h,
                                                       gru_model.weight_ih_l0,
                                                       gru_model.weight_hh_l0,
                                                       gru_model.bias_ih_l0,
                                                       gru_model.bias_hh_l0)[1]
            target_tree_list.append(target_tree)
        
#        gru_h_list = gru_h_list[:, 1:, :]
#        gru_loss1 = 0
#        for i_time in range(y.shape[1]):
#            gru_loss1 += loss(model.output_layer(gru_h_list[:, i_time, :]), torch.argmax(y[:, i_time, :], dim=1))
            
        # get weight for each loss terms
        lamb1, lamb2, lamb3, lamb4 = params['lambdas']
        
        # calculate loss terms
        loss1 = 0
        if lamb1 != 0:
            for i_time in range(y.shape[1]):
                loss1 += loss(pred_chars_batch[:, i_time, :], torch.argmax(y[:, i_time, :], dim=1))

        loss2 = 0
        if lamb2 != 0:
            desired_margin = params['loss2_margin']
            loss2 = (desired_margin - margins_batch.clamp(max=desired_margin)).sum()/desired_margin
            
        loss3 = 0
        if lamb3 != 0:
            for param in model.parameters():
                loss3 += param.norm()**2
        
        loss4 = 0
        if lamb4 != 0:
            for i_time_step in range(time_steps):
                loss4 += tree_methods.tree_distance_metric_list(
                                            pred_tree_list[i_time_step], 
                                            target_tree_list[i_time_step])

            
#        losses = [lamb1*loss1]
#        losses = [lamb1*loss1, lamb3*loss3]
#        losses = [lamb1*loss1, lamb3*loss3, lamb4*loss4]
        losses = [lamb1*loss1, lamb2*loss2, lamb3*loss3, lamb4*loss4]

        accuracy = (pred_chars_batch.argmax(dim=2)==y.argmax(dim=2)).sum().item()/float(time_steps*batch_size)
        
        
        # opt
        loss_fn = sum(losses)
        loss_fn.backward()
        optimizer.step()
        print('='*80)
        print(i_batch, time.time()-timer)
        print(i_epoch, i_batch, loss_fn.item(), time.time()-timer)
        print(losses)
        print(accuracy, -np.log2(accuracy), loss1.item()/(20*np.log(2)))
        # target bpc1: 1.17, bpc2: 2.73
    











