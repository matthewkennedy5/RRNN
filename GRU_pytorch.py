# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 22:00:00 2018

@author: Bruce
"""


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



timer = time.time()
import dataloader
X_train, y_train = dataloader.load_data('train20.txt')
tmp = torch.cat(X_train, dim=1).reshape(-1, 100).numpy()
tmp_mean = np.mean(tmp, axis=0) 
tmp_std = np.std(tmp, axis=0)
for i in range(len(X_train)):
    X_train[i] = (X_train[i]-torch.from_numpy(tmp_mean))/torch.from_numpy(tmp_std)


nb_epochs = 100


class GRUTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(GRUTagger, self).__init__()
        self.hidden_dim = hidden_dim

#        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return torch.zeros(1, self.hidden_dim, requires_grad=True) 

    def forward(self, X):
#        embeds = self.word_embeddings(sentence)
        gru_out, self.hidden = self.gru(X)
        tag_space = self.hidden2tag(self.hidden)
        tag_scores = F.softmax(tag_space.reshape(1, 27))
        return tag_scores


model = GRUTagger(100, 100, 27, 27)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters())

loss = torch.nn.CrossEntropyLoss()
#loss = torch.nn.KLDivLoss(size_average=False)
    

# todo
# 还没有把gru的真实code放在loss function里面，现在算出来的loss还是不对的


loss_list = []
for e in range(nb_epochs):
    for i in range(len(X_train)):
        X = X_train[i]
        y = y_train[i]
        target = torch.zeros(1, len(y))
        target[0, np.argmax(y)] = 1
        # set to training mode
        model.train()
        
         # zero gradient
        optimizer.zero_grad()
        
        # forward pass and compute loss
        out = model(X)

        # compute gradient and take step in optimizer
        loss_fn = -(torch.log(out)*target).sum()
        loss_fn.backward() 
#        break

        optimizer.step()
        
        loss_list.append(loss_fn.data.tolist())
        if len(loss_list) > 10000:
            loss_list = loss_list[-10000:]
        if i%100 == 0:
            print('Epoch:', e+1, i, loss_fn, np.mean(loss_list[-100:]), time.time()-timer)
            print('='*80)
            file=open('gru_log.txt', 'a')
            file.write(str(np.mean(loss_list[-100:]))+'\n')
            file.close()
            
model.eval() # set to evaluation mode
torch.save(model.gru, 'gru_parameters.pkl')