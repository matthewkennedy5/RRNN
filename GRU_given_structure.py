#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:25:11 2019

@author: Bruce
"""



import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import uniform
import numpy as np
import time

from tree_methods import Node
import tree_methods


device = torch.device('cpu')


def retrieve_binary_operation(s):
    if s == 'add':
        binary = lambda x,y: x+y
    elif s == 'mul':
        binary = lambda x,y: x*y
    else:
        raise ValueError('No such binary function %s!'%s)
    return binary

def retrieve_unary_operation(s):
    if s == 'sigmoid':
        unary = lambda x: torch.sigmoid(x)
    elif s == 'tanh':
        unary = lambda x: torch.tanh(x)
    elif s == 'minus':
        unary = lambda x: 1-x
    elif s == 'identity':
        unary = lambda x: x    
    else:
        raise ValueError('No such unary function %s!'%s)
    return unary

def retrieve_node(string, x, h_prev, G_node):
    '''
        return a Node instance with given vector
    '''
    if string == 'x':
        return Node(x, 'x')
    elif string == 'h_prev':
        return Node(h_prev, 'h_prev')
    elif string == 'o':
        return Node(torch.zeros(x.shape, device=device), 'o')
    elif string[0] == 'G':
        return G_node[int(string[2:-1])]
    else:
        raise ValueError

def margin_loss(scores):
    """Returns a Tensor of the difference between the top two scores.
    Input:
        scores - a list of outputs from the scoring net. Must contain at
            least two elements.
    """
    sorted_indices = torch.sort(torch.Tensor(scores))[1]
    highest = sorted_indices[-1]
    second_highest = sorted_indices[-2]
    return scores[highest] - scores[second_highest]


class RRNNforGRUCell(nn.Module):
    def __init__(self, hidden_size, scoring_hsize=None):
        super(RRNNforGRUCell, self).__init__()

        self.hidden_size = hidden_size
        self.binary_ops_list = ['mul', 'add']
        self.m = 1  # Num of output vectors
        self.N = 9  # Num of generated nodes in one cell
        self.l = 4  # Num of parameter matrices (L_i, R_i, b_i)

        # Initalize L R weights from the uniform distribution
        weight_distribution = uniform.Uniform(-1/np.sqrt(hidden_size), 1/np.sqrt(hidden_size))
        Ls = []
        Rs = []
        for i in range(self.l):
            L = nn.Parameter(weight_distribution.sample(sample_shape=torch.zeros([hidden_size, hidden_size]).shape))
            R = nn.Parameter(weight_distribution.sample(sample_shape=torch.zeros([hidden_size, hidden_size]).shape))
            Ls.append(L)
            Rs.append(R)
        self.L_list = nn.ParameterList(Ls)
        self.R_list = nn.ParameterList(Rs)
        # The bias terms are initialized as zeros.
        self.b_list = nn.ParameterList([nn.Parameter(torch.zeros(1, hidden_size)) for _ in range(self.l)])

        #################################################
        # Initilize L3, R3, b3 to the identity matrices #
        self.L_list[3] = nn.Parameter(torch.eye(hidden_size))
        self.R_list[3] = nn.Parameter(torch.eye(hidden_size))
        self.b_list[3] = nn.Parameter(torch.zeros(1, hidden_size))
        ## Freeze them - they won't train to something else
        #self.L_list[3].requires_grad = False
        #self.R_list[3].requires_grad = False
        #self.b_list[3].requires_grad = False
        #################################################

        if scoring_hsize is not None:
            self.scoring = nn.Sequential(
                                nn.Linear(hidden_size, scoring_hsize),
                                nn.ReLU(),
                                nn.Linear(scoring_hsize, 1)
                           )
        else:
            self.scoring = nn.Linear(hidden_size, 1, bias=False)
            
    def forward(self, x, h_prev, pred_structure):
        # =============================================================================
        # structure info should be a list of 5 elements. the first two elements are left and
        # right childs, exactly 'x', 'h_prev', 'o' (stands for zero vector), or 'G[k]'.
        # The third element should be the binary operation whild the 4th element 
        # should be the activation function. The last one is the index of parameter 
        # matrices used to comstruct this node. Samples are 
        # ['x', 'h_prev', 'add', 'sigmoid', 3]
        # ['G[0]', 'G[2]', 'mul', 'identity', 3]
        #
        # I hard coded the pred_structure here to be exactly GRU (ignoring the 
        # last element, which will not be used), you have to remove the following 
        # dict later.
        # =============================================================================

        pred_structure = {
            0: ['x', 'h_prev', 'add', 'sigmoid', 1],   # z1
            1: ['x', 'h_prev',  'add', 'sigmoid', 1],   # r
            2: ['x', 'h_prev',  'add', 'sigmoid', 1],   # z2
            3: ['G[1]', 'h_prev',  'mul', 'identity', 2], # r*h
            4: ['G[3]', 'x',  'add', 'tanh', 0], # h_tilde
            5: ['G[0]', 'o',  'add', 'minus', 2],    # 1-z
            6: ['G[4]', 'G[5]',  'mul', 'identity',3 ],    # (1-z)*h_tilde
            7: ['G[2]', 'h_prev',  'mul', 'identity', 3], # z*h
            8: ['G[6]', 'G[7]', 'add', 'identity', 3]    # h_t}
        }
        
        l = self.l
        o = torch.zeros(1, self.hidden_size)
        softmax_func = torch.nn.Softmax(dim=0)
        G = []
        G_structure = []
        G_node = []
        G_margin = []
        
        for i_node in range(self.N): # len(structure) == self.N

            structure = pred_structure[i_node]
            left_vector = eval(structure[0])            
            right_vector = eval(structure[1])            
            binary = retrieve_binary_operation(structure[2])
            unary = retrieve_unary_operation(structure[3])
            
            # generate potential choice of vectors for i-th node
            lst = []
            for k in range(l):
                lst.append(unary(binary(torch.mm(left_vector, self.L_list[k]),
                                        torch.mm(right_vector, self.R_list[k]))
                                 +self.b_list[k]))
            scores_lst = [self.scoring(v).squeeze() for v in lst]
            structure = structure[:4]+[np.argmax(scores_lst)]
            softmax_vector = sum([softmax_func(torch.Tensor(scores_lst))[i]*lst[i] for i in range(l)])
            
            print(structure)
            left_node = retrieve_node(structure[0], x, h_prev, G_node)
            right_node = retrieve_node(structure[1], x, h_prev, G_node)
            node = Node(softmax_vector, name='G[%d]'%i_node, structure=structure, left_child=left_node, right_child=right_node)
            
            G.append(softmax_vector)
            G_structure.append(structure)
            G_node.append(node)
            G_margin.append(margin_loss(scores_lst))
            
        for node in G_node:
            node.leftchild.parent = node
            node.rightchild.parent = node
            
        h_next = G[-1]
        
        return h_next, G_structure, G_node, sum(G_margin)


class RRNNforGRU(nn.Module):
    def __init__(self, hidden_size, vocab_size, scoring_hsize=None):
        super(RRNNforGRU, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.cell = RRNNforGRUCell(hidden_size, scoring_hsize=scoring_hsize)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size, requires_grad=True, device=device)

    def forward(self, inputs, pred_structures):
        time_steps = inputs.size(1)
        h_next = self.init_hidden()
        h_list = []
        pred_tree_list = []
        margin_list = []
        structure_list = []
        pred_chars_list = []

        for t in range(time_steps):
            x_t = inputs[:, t, :].reshape(1, -1)
            pred_structure = pred_structures[t]
            h_next, G_structure, G_node, margin = self.cell(x_t, h_next, pred_structure)
            pred_chars_list.append(self.output_layer(h_next))
            h_list.append(h_next)
            structure_list.append(G_structure)
            pred_tree_list.append(G_node)
            margin_list.append(margin)

        # note that all following list have length 20
        return pred_chars_list, h_list, structure_list, pred_tree_list, margin_list
        
if __name__ == '__main__':
    # test case for RRNNforGRUCell
    x = torch.randn(1, 5)
    hidden = torch.randn(1, 5)
    
    cell = RRNNforGRUCell(5)
    h_next, G_structure, G_node, margin = cell(x, hidden, 1)
    
    # test case for RRNNforGRU
    inputs = torch.randn(1, 3, 5)
    model = RRNNforGRU(hidden_size=5, vocab_size=27, scoring_hsize=50)
    cell = model.cell
    pred_chars_list, h_list, structure_list, pred_tree_list, margin_list = model(inputs, [[],[],[]])
    
            
            
            
            
            
            
            
            
            
            
            
            
            