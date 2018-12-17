# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 02:16:41 2018

@author: Bruce
"""

import torch
from gensim.models import Word2Vec 
word2vec_model = Word2Vec.load('train20_embedding')

element_dict = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,'k':10
            ,'l':11,'m':12,'n':13,'o':14,'p':15,'q':16,'r':17,'s':18,'t':19,
            'u':20,'v':21,'w':22,'x':23,'y':24,'z':25,' ':26}


def load_data(filename):
    X_train = []
    y_train = []
    file = open(filename, 'r')
    for i in range(5000):
        input_word = file.readline()[:-1]
        word = []
        for j in range(len(input_word)-1):
            if input_word[j] == ' ':
                word.append(word2vec_model['.'])
            else:
                word.append(word2vec_model[input_word[j]])
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

if __name__ == '__main__':
    X_train, y_train = load_data('train20.txt')