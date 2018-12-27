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
            if embeddings == 'word2vec':
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


def load_normalized_data(filename, embeddings='gensim'):
    X_train, y_train = load_data(filename, embeddings)
    tmp = torch.cat(X_train, dim=1).reshape(-1, 100).numpy()
    tmp_mean = np.mean(tmp, axis=0)
    tmp_std = np.std(tmp, axis=0)
    for i in range(len(X_train)):
        X_train[i] = (X_train[i]-torch.from_numpy(tmp_mean))/torch.from_numpy(tmp_std)
    return X_train, y_train


if __name__ == '__main__':
    X_train, y_train = load_data('train20.txt')