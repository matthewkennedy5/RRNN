# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 02:16:41 2018

@author: Bruce
"""

import torch
from gensim.models import Word2Vec
import numpy as np
import load_embeddings

word2vec_model = Word2Vec.load('train20_embedding')

element_dict = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,'k':10
            ,'l':11,'m':12,'n':13,'o':14,'p':15,'q':16,'r':17,'s':18,'t':19,
            'u':20,'v':21,'w':22,'x':23,'y':24,'z':25,' ':26}
MAGIC_EMBEDDINGS_FILE = 'char-embeddings.txt'


def load_data(filename, embeddings='gensim'):
    X_train = []
    y_train = []
    file = open(filename, 'r')
    if embeddings == 'magic':
        magic_embeddings = load_embeddings.load(MAGIC_EMBEDDINGS_FILE)

    for i in range(5000):
        input_word = file.readline()[:-1]   # Word means a line of text
        word = []   # List of embedded character vectors for the line
        for j in range(len(input_word)-1):
            if embeddings == 'gensim':
                if input_word[j] == ' ':
                    word.append(word2vec_model['.'])
                else:
                    word.append(word2vec_model[input_word[j]])
            elif embeddings == 'magic':
                word.append(magic_embeddings[input_word[j]])
            else:
                raise ValueError('embeddings must be either "word2vec" or "magic"')
#
#            element_vector = [0]*27
#            element_vector[element_dict[input_word[j]]]=1
#            word.append(element_vector)
        X_train.append(torch.tensor(word).reshape(1, -1, 100))
        y = [0 for i in range(len(element_dict.keys()))]
        y[element_dict[input_word[-1]]] = 1
        y_train.append(y)
#        y_train.append(torch.tensor(word2vec_model[input_word[-1]]))
#        y_train.append(torch.tensor([element_dict[input_word[-1]]]))
    file.close()
    return X_train, y_train


def load_normalized_data(filename, n_train, n_val, device, embeddings='gensim', shuffle=True):
    """Loads a version of the dataset that has been normalized.

    Inputs:
        filename - Name of a file containing rows of text, the last character of
            which will be used as the truth.
        n_train - Desired number of training sequences.
        n_val - Desired number of validation sequences.
        device - torch.device to put the tensors on
        embeddings - 'gensim' for 100-dimensional or 'magic' for 300-dimensional
        shuffle - Whether to shuffle the training data

    Returns:
        X_train - Training x tensor
        y_train - Training y tensor (one-hot)
        X_val - Validation data
        y_val -
    """
    X_train, y_train = load_data(filename, embeddings)
    X_val = X_train[-n_val:]    # Split of the last n_val examples for validation
    y_val = y_train[-n_val:]
    X_train = X_train[:-n_val]  # Remove the validation examples from the training
    y_train = y_train[:-n_val]

    tmp = torch.cat(X_train, dim=1).reshape(-1, 100).numpy()
    tmp_mean = np.mean(tmp, axis=0)
    tmp_std = np.std(tmp, axis=0)
    for i in range(len(X_train)):
        X_train[i] = (X_train[i]-torch.from_numpy(tmp_mean))/torch.from_numpy(tmp_std)
    for i in range(len(X_val)):
        X_val[i] = (X_val[i] - torch.from_numpy(tmp_mean)) / torch.from_numpy(tmp_std)

    # Convert X and y from lists to tensors of rank 4 and put them on the proper device
    for i in range(len(X_train)):
        X_train[i] = X_train[i].to(device)
        y_train[i] = torch.tensor(y_train[i], device=device)

    for i in range(len(X_val)):
        X_val[i] = X_val[i].to(device)
        y_val[i] = torch.tensor(y_val[i], device=device)

    X_train = torch.stack(X_train, dim=0)
    y_train = torch.stack(y_train, dim=0)
    X_val = torch.stack(X_val, dim=0)
    y_val = torch.stack(y_val, dim=0)

    # Shuffle X_train and y_train in the same way
    if shuffle:
        indices = np.random.choice(range(len(X_train)), size=n_train, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]

    return X_train, y_train, X_val, y_val


if __name__ == '__main__':
    X_train, y_train = load_data('train20.txt')

