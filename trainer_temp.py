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
import itertools

# import tree_methods_parallel as tree_methods
import tree_methods_accelerated as tree_methods
from GRU_parallel import RRNNforGRU
from structure_utils import structures_are_equal, GRU_STRUCTURE
import pickle
import pdb
from tqdm import tqdm, trange
import standard_data

VOCAB_SIZE = 27
HIDDEN_SIZE = 100
batch_size = 64
n_epoch = 1

device = torch.device('cpu')

TRAIN_LOSS_FILE = 'loss.txt'
TRAIN_ACC_FILE = 'train_acc.txt'
VAL_LOSS_FILE = 'val_loss.txt'
VAL_ACC_FILE = 'val_acc.txt'
HYPERPARAM_FILE = 'hyperparameters.pkl'
RUNTIME_FILE = 'runtime.pkl'

params = {
    'learning_rate': 1e-4,
    'multiplier': 1,
    'lambdas': (1, 1, 1, 1),
    'nb_train': 10,   # Only meaningful if it's less than the training set size
    'nb_val': 10,
    # TODO: Make this epochs
    'validate_every': 1,  # How often to evaluate the validation set (iterations)
    'epochs': 1,
    'n_processes': mp.cpu_count(),
    'loss2_margin': 1,
    'scoring_hidden_size': 64,     # Set to None for no hidden layer
    'batch_size': 16,
    'verbose': True,
    'epochs_per_checkpoint': 1,
    'optimizer': 'adam',
    'debug': False,  # Turns multiprocessing off so pdb works
    'data_file': 'enwik8_clean.txt',
    'embeddings': 'gensim',
    'max_grad': 1,  # Max norm of gradients. Set to None for no clipping
    'initial_train_mode': 'weights',
    'alternate_every': 5,    # Switch training mode after this many epochs
    'warm_start': False,
    'weights_file': 'epoch_0.pt'
}

dirname = 'test%s'%(time.asctime().replace(':', '_'))
if not params['warm_start']:
    os.mkdir(dirname)
os.chdir(dirname)

gru_model = torch.load('../gru_parameters.pkl')
model = RRNNforGRU(HIDDEN_SIZE, VOCAB_SIZE, batch_size, params['scoring_hidden_size'])
filename = os.path.join('..', params['data_file'])
data = standard_data.load_standard_data()
(X_train, y_train), (X_val, y_val), (X_test, y_test) = data
X_train = X_train[:4096, :, :]
y_train = y_train[:4096, :, :]

if params['optimizer'] == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
elif params['optimizer'] == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'])

lamb1, lamb2, lamb3, lamb4 = params['lambdas']
loss = torch.nn.CrossEntropyLoss()
#
#CUDA = True
#if CUDA is True:
#    X_train = X_train.cuda()
#    y_train = y_train.cuda()
#    model = model.cuda()
#    gru_model = gru_model.cuda()

model.train()

timer=time.time()
for i_epoch in range(n_epoch):
    for i_batch in range(X_train.shape[0]//batch_size):
        X = X_train[i_batch*batch_size:(i_batch+1)*batch_size, :, :]
        y = y_train[i_batch*batch_size:(i_batch+1)*batch_size, :, :]
        
        # forward pass
        pred_chars_batch, h_batch, pred_tree_list, structures_list, margins_batch = model(X)
        
        # forward pass of traditional GRU
        gru_h_list = gru_model(X)[0]
        gru_h_list = torch.cat([torch.zeros(batch_size, 1, HIDDEN_SIZE), gru_h_list], dim=1)
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
        
        # calculate loss function
        loss1 = 0
        if lamb1 != 0:
            for i_time in range(y.shape[1]):
                loss1 += loss(pred_chars_batch[:, i_time, :], torch.argmax(y[:, i_time, :], dim=1))
        
        # loss2 is the negative sum of the scores (alpha) of the vector
        # corresponding to each node. It is an attempt to drive up the scores for
        # the correct vectors.
#        loss2 = 0
#        if lamb2 != 0:
#            desired_margin = params['loss2_margin']
#            for m in margins:
#                if m < desired_margin:
#                    # Here the subtraction comes from the fact that we want the
#                    # loss to be 0 when the difference >= LOSS2_MARGIN,
#                    # and equal to 1 when the difference is 0. Therefore,
#                    # loss2 will always be between 0 and the number of
#                    # vectors we have. We divide by LOSS2_MARGIN to scale
#                    # the loss term to be between 0 and 1, so it LOSS2_MARGIN
#                    # doesn't affect the overall scale of loss2.
#                    value = torch.clamp(m, min=0) / desired_margin
#                    if value > 0:
#                        loss2 += value
        
        loss3 = 0
        if lamb3 != 0:
            for param in model.parameters():
                loss3 += param.norm()**2
        
        loss4 = 0
        if lamb4 != 0:
            for l in range(len(pred_tree_list)):
                loss4 += tree_methods.tree_distance_metric_list(pred_tree_list[l],
                                                                target_tree_list[l])
        
        losses = (lamb1*loss1, lamb3*loss3, lamb4*loss4)
        loss_fn = sum(losses)
        
        # opt
        loss_fn.backward()
        optimizer.step()
        
        print(i_batch, loss_fn.item(), time.time()-timer)
        











