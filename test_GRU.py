# -*- coding: utf-8 -*-


from GRU import RRNNforGRUCell, RRNNforGRU, _cuda

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Module, Parameter
from torch.autograd import Function

from tree_methods import Node
import tree_methods

import numpy as np
import time


# load pretrained GRU model
gru_model = torch.load('gru_parameters.pkl')
Lr, Lz, Lh = gru_model.weight_ih_l0.chunk(3)
Rr, Rz, Rh = gru_model.weight_hh_l0.chunk(3)
b_ir, b_iz, b_in = gru_model.bias_ih_l0.chunk(3)
b_hr, b_hz, b_hn = gru_model.bias_hh_l0.chunk(3)
br = b_ir + b_hr
bz = b_iz + b_hz


timer = time.time()
import dataloader
X_train, y_train = dataloader.load_data('train20.txt')

_hidden_size = 100
_vocab_size = 27
nb_epochs = 5

model = RRNNforGRU(_hidden_size, _vocab_size)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
#loss = torch.nn.CrossEntropyLoss()
loss = torch.nn.KLDivLoss()


#_cuda = GRU._cuda
if _cuda is True:
    for i in range(5000):
        X_train[i] = X_train[i].cuda()
        y_train[i] = y_train[i].cuda()
    model = model.cuda()


lamb1 = 1
lamb3 = 0
lamb4 = 1

for e in range(nb_epochs):
    for i in range(len(X_train)):
        X = X_train[i]
        y = y_train[i]
        # set to training mode
        model.train()

         # zero gradient
        optimizer.zero_grad()

        # forward pass and compute loss
        out, h_list, pred_tree_list = model(X)

        # forward pass of traditional GRU
        gru_h_list = gru_model(X)[0]
        gru_h_list = torch.cat([torch.zeros(1,1,_hidden_size), gru_h_list], dim=1)
        target_tree_list = []
        for t in range(X.shape[1]):
            gru_x = X[:, t, :]
            gru_h = gru_h_list[:, t, :]
            target_tree = tree_methods.GRUtree_pytorch(gru_x, gru_h, gru_model.weight_ih_l0, gru_model.weight_hh_l0, gru_model.bias_ih_l0, gru_model.bias_hh_l0)[1]
            target_tree_list.append(target_tree)

        # calculate loss function
        loss1 = 0
        if lamb1 != 0:
            loss1 = loss(out, torch.Tensor(y).reshape(1,27))

        loss3 = 0
        if lamb3 != 0:
            for param in model.parameters():
                loss3 += param.norm()

        loss4 = 0
        if lamb4 != 0:
            for l in range(len(pred_tree_list)):
                loss4 += tree_methods.tree_distance_metric_list(pred_tree_list[l], target_tree_list[l])

        # compute gradient and take step in optimizer
        loss_fn = lamb1*loss1 + lamb3*loss3 + lamb4*loss4
        loss_fn.backward()

        optimizer.step()

        print('Epoch:', e+1, i, loss_fn, time.time()-timer)
        print(model.cell.L_list[0])
        print('='*80)

model.eval() # set to evaluation mode

