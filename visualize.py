#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:37:33 2019

@author: Bruce
"""

import os
import pickle 
import matplotlib.pyplot as plt
import pandas as pd

dirname = 'batch10'
os.chdir(dirname)

params = pickle.load(open('hyperparameters.pkl', 'rb'))
for key, value in params.items():
    print(key, ':', value)
    
train_loss_filename = 'train_loss.txt'
train_acc_filename = 'train_acc.txt'
val_loss_filename = 'val_loss.txt'
val_acc_filename = 'val_acc.txt'

acc_cols = ['epoch', 'batch', 'stage', 'acc']
loss_cols = ['epoch', 'batch', 'stage', 'loss1', 'loss2', 'loss3', 'loss4']
acc_cols = ['epoch', 'batch', 'acc']
loss_cols = ['epoch', 'batch', 'loss1', 'loss2', 'loss3', 'loss4']

train_loss = pd.read_csv(train_loss_filename, sep=';', header=None, names=loss_cols)
val_loss = pd.read_csv(val_loss_filename, sep=';', header=None, names=loss_cols)
train_acc = pd.read_csv(train_acc_filename, sep=';', header=None, names=acc_cols)
val_acc = pd.read_csv(val_acc_filename, sep=';', header=None, names=acc_cols)

val_loss.groupby('epoch').mean()['loss1'].plot()
