# -*- coding: utf-8 -*-


from GRU import RRNNforGRUCell, RRNNforGRU, device

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Module, Parameter
from torch.autograd import Function

from tree_methods import Node
import tree_methods

from matplotlib import pyplot as plt
import numpy as np
import pickle
import time
from progressbar_utils import init_progress_bar
import dataloader

LOSS_FILE = 'loss3.pkl'
SAVE_FILE = 'loss-plots/13.png'

# Hyperparameters
LEARNING_RATE = 1e-3
lamb1 = 1   # Controls the loss for the output character
lamb2 = 0   # Scoring loss
lamb3 = 0   # L2 regularization loss
lamb4 = 0   # Tree distance loss
nb_epochs = 1000
NB_DATA = 1
EMBEDDINGS = 'magic'    # 'magic' or 'gensim'

# load pretrained GRU model
gru_model = torch.load('gru_parameters.pkl').to(device)
Lr, Lz, Lh = gru_model.weight_ih_l0.chunk(3)
Rr, Rz, Rh = gru_model.weight_hh_l0.chunk(3)
b_ir, b_iz, b_in = gru_model.bias_ih_l0.chunk(3)
b_hr, b_hz, b_hn = gru_model.bias_hh_l0.chunk(3)
br = b_ir + b_hr
bz = b_iz + b_hz

timer = time.time()
X_train, y_train = dataloader.load_normalized_data('train20.txt',
                                                   embeddings=EMBEDDINGS)
X_train = X_train[:NB_DATA]

_hidden_size = 100
_vocab_size = 27

model = RRNNforGRU(_hidden_size, _vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#loss = torch.nn.CrossEntropyLoss()
loss = torch.nn.KLDivLoss()

#_cuda = GRU._cuda
# if _cuda is True:
for i in range(len(X_train)):
    X_train[i] = torch.tensor(X_train[i], device=device)
    y_train[i] = torch.tensor(y_train[i], device=device)
model = model.to(device)

loss_history = np.zeros(nb_epochs * len(X_train))
bar = init_progress_bar(nb_epochs * len(X_train))
bar.start()
for e in range(nb_epochs):
    for i in range(len(X_train)):
        X = X_train[i]
        y = y_train[i]
        # set to training mode
        model.train()

         # zero gradient
        optimizer.zero_grad()

        # forward pass and compute loss
        out, h_list, pred_tree_list, scores, second_scores = model(X)

        # forward pass of traditional GRU
        gru_h_list = gru_model(X)[0].to(device)
        gru_h_list = torch.cat([torch.zeros(1,1,_hidden_size, device=device), gru_h_list], dim=1)
        target_tree_list = []
        for t in range(X.shape[1]):
            gru_x = X[:, t, :]
            gru_h = gru_h_list[:, t, :]
            target_tree = tree_methods.GRUtree_pytorch(gru_x, gru_h,
                                                       gru_model.weight_ih_l0,
                                                       gru_model.weight_hh_l0,
                                                       gru_model.bias_ih_l0,
                                                       gru_model.bias_hh_l0)[1]
            target_tree_list.append(target_tree)

        # calculate loss function
        loss1 = 0
        if lamb1 != 0:
            loss1 = loss(out, y.reshape(1,27).float())

        # loss2 is the negative sum of the scores (alpha) of the vector
        # corresponding to each node. It is an attempt to drive up the scores for
        # the correct vectors.
        loss2 = 0
        if lamb2 != 0:
            loss2 = -np.sum(scores) + np.sum(second_scores)

        loss3 = 0
        if lamb3 != 0:
            for param in model.parameters():
                loss3 += param.norm()

        loss4 = 0
        if lamb4 != 0:
            for l in range(len(pred_tree_list)):
                loss4 += tree_methods.tree_distance_metric_list(pred_tree_list[l], target_tree_list[l], device=device)

        # compute gradient and take step in optimizer
        loss_fn = lamb1*loss1 + lamb2*loss2 + lamb3*loss3 + lamb4*loss4
        loss_fn.backward()

        optimizer.step()

        print('Epoch:', e+1, i, loss_fn, time.time()-timer)
        print(model.cell.L_list[0])
        print('='*80)
        index = e * len(X_train) + i
        loss_history[index] = loss_fn.item()
        bar.update(index + 1)
        if index % 100 == 0:    # Save out the loss as we go
            pickle.dump(loss_history, open(LOSS_FILE, 'wb'))
            print('\n[INFO] Saved loss history.')

model.eval() # set to evaluation mode

plt.figure()
plt.plot(range(loss_history.shape[0]), loss_history)
plt.savefig(SAVE_FILE)
