# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:15:54 2019

@author: Bruce
"""

import torch
import torch.nn as nn
from torch.distributions import uniform
import numpy as np
import time
import pdb

from tree_methods import Node


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
    elif s == 'relu':
        unary = lambda x: torch.relu(x)
    else:
        raise ValueError('No such unary function %s!'%s)
    return unary

def margins_func(scores):
    sorted_scores = torch.sort(scores)[0]
    return sorted_scores[:, :, -1] - sorted_scores[:, :, -2]


class RRNNforGRUCell(nn.Module):
    def __init__(self, hidden_size, scoring_hsize=None):
        super(RRNNforGRUCell, self).__init__()

        self.hidden_size = hidden_size
        self.binary_ops_list = ['add', 'mul']
        self.unary_ops_list = ['sigmoid', 'tanh', 'minus', 'identity', 'relu']
        self.m = 1  # Num of output vectors
        self.N = 9  # Num of generated nodes in one cell
        self.l = 4  # Num of parameter matrices (L_i, R_i, b_i)

        # Initalize L R weights from the uniform distribution
        weight_distribution = uniform.Uniform(-1/np.sqrt(hidden_size), 1/np.sqrt(hidden_size))
        Ls = []
        Rs = []
        for i in range(self.l):
            L = nn.Parameter(weight_distribution.sample([hidden_size, hidden_size]))
            R = nn.Parameter(weight_distribution.sample([hidden_size, hidden_size]))
            Ls.append(L)
            Rs.append(R)

        self.L_list = nn.ParameterList(Ls)
        self.R_list = nn.ParameterList(Rs)
        # The bias terms are initialized as zeros.
        self.b_list = nn.ParameterList([nn.Parameter(torch.zeros(1, hidden_size)) for _ in range(self.l)])

        #################################################
        # Set L3, R3, b3 to the identity transformation #
        self.L_list[3] = nn.Parameter(torch.eye(hidden_size))
        self.R_list[3] = nn.Parameter(torch.eye(hidden_size))
        self.b_list[3] = nn.Parameter(torch.zeros(1, hidden_size))
        self.L_list[3].requires_grad = True
        self.R_list[3].requires_grad = True
        self.b_list[3].requires_grad = True
        #################################################
        
        self.batchnorm = nn.BatchNorm1d(self.hidden_size, affine=False, track_running_stats=False)
        if scoring_hsize is not None:
            self.scoring = nn.Sequential(
                                nn.Linear(hidden_size, scoring_hsize),
                                nn.ReLU(),
                                nn.Linear(scoring_hsize, 1)
                           )
        else:
            self.scoring = nn.Linear(hidden_size, 1, bias=False)
    
    
    def forward_stage_search(self, x, h_prev):
        batch_size, _, hidden_size = x.shape
        o = torch.zeros_like(x, device=x.device)
        l = self.l
        softmax_func = torch.nn.Softmax(dim=2)
        
        G = []
        G_node = []
        G_structure = []
        G_margin = []
        components_list = []
             
        for r in range(self.N):
            candidate = [x, h_prev, o] + [comp.vector for comp in components_list]
            L_times_left_child = [[torch.matmul(candidate[i], self.L_list[k]) for k in range(l)] for i in range(len(candidate))]
            R_times_left_child = [[torch.matmul(candidate[j], self.R_list[k]) for k in range(l)] for j in range(len(candidate))]
            V = []
            V_structure = []
                        
            for i in range(len(candidate)-1):
                for j in range(i+1, len(candidate)):
                    flag = 0
                    if (self.N - r) == len(components_list) - 1: # You could only merge two components
                        if i >= 3:
                            flag = 1
                    elif (self.N - r) == len(components_list): # could merge or keep # components to be the same
                        if j >= 3:
                            flag = 1
                    elif (self.N - r) > len(components_list):  # whatever you want
                        flag = 1
                    else:
                        raise ValueError
                    
                    if flag == 1:
                        if r == 0:
                            available_LR_index = [0]
                        elif r == 1:
                            available_LR_index = [1]
                        else:
                            available_LR_index = list(range(l))
                        
                        for k in available_LR_index:
                            b = self.b_list[k]
                            for binary_func in self.binary_ops_list:
                                if binary_func == 'add':
                                    res = L_times_left_child[i][k] + R_times_left_child[j][k] + b
                                else:   # elif binary_func == 'mul':
                                    res = L_times_left_child[i][k] * R_times_left_child[j][k] + b
                                if res.abs().max() > 1:
                                    V.append(torch.sigmoid(res))
                                    V_structure.append([i, j, k, binary_func, 'sigmoid'])
                                    V.append(torch.tanh(res))
                                    V_structure.append([i, j, k, binary_func, 'tanh'])                                    
                                else:
                                    V.append(torch.sigmoid(res))
                                    V_structure.append([i, j, k, binary_func, 'sigmoid'])
                                    V.append(torch.tanh(res))
                                    V_structure.append([i, j, k, binary_func, 'tanh'])
                                    V.append(1-res)
                                    V_structure.append([i, j, k, binary_func, 'minus'])
                                    V.append(res)
                                    V_structure.append([i, j, k, binary_func, 'identity'])
                                    V.append(torch.relu(res))
                                    V_structure.append([i, j, k, binary_func, 'relu'])                   
            
            V = torch.cat(V, dim=1)
            
            V = self.batchnorm(V)
#            V = (V-V.mean(dim=0))/(1e-5+V.std(dim=0))
            scores = self.scoring(V).reshape(batch_size, 1, -1)
            weighted_vector = torch.matmul(softmax_func(scores), V)
            margin = margins_func(scores)
            structure = V_structure[torch.argmax(scores.sum(dim=0)).item()]
            
            G.append(weighted_vector)
            G_margin.append(margin)
                        
            i, j = structure[:2]
            name_list = ['x', 'h_prev', 'o'] + ['G[%d]'%i for i in range(r)]
            if i>2 and j>2: # merged two components
                left_node = components_list[i-3]
                right_node = components_list[j-3]
                new_component = Node(weighted_vector, name='G[%d]'%r, 
                                     structure=[left_node.name, right_node.name]+structure[2:],
                                     left_child=left_node, right_child=right_node)
                components_list[i-3] = new_component
                del components_list[j-3]
            elif i<=2 and j>2:
                left_node = Node(candidate[i], name=name_list[i])
                right_node = components_list[j-3]
                new_component = Node(weighted_vector, name='G[%d]'%r, 
                                     structure=[left_node.name, right_node.name]+structure[2:],
                                     left_child=left_node, right_child=right_node)
                components_list[j-3] = new_component
            else:   # elif i<=2 and j<=2:
                left_node = Node(candidate[i], name=name_list[i])
                right_node = Node(candidate[j], name=name_list[j])
                new_component = Node(weighted_vector, name='G[%d]'%r, 
                                     structure=[left_node.name, right_node.name]+structure[2:],
                                     left_child=left_node, right_child=right_node)                
                components_list.append(new_component)
            
            G_structure.append(new_component.structure)
            G_node.append(new_component)
        
        G_margin = torch.cat(G_margin, dim=1)
        return G[-1], G_structure, G_node, G_margin
        
    def foward_stage_fixing(self, x, h_prev, G_structure):
        o = torch.zeros_like(x, device=x.device)
        G = []
    
        for structure in G_structure:
            left_name, right_name, k, binary_func, unary_func = structure
            unary_func = retrieve_unary_operation(unary_func)
            binary_func = retrieve_binary_operation(binary_func)
            L, R, b = self.L_list[k], self.R_list[k], self.b_list[k]
            v = unary_func(binary_func(torch.matmul(eval(left_name), L), torch.matmul(eval(right_name), R))+b)
            G.append(v)
        
        return v
    
    
class RRNNforGRU(nn.Module):
    def __init__(self, hidden_size, vocab_size, batch_size, scoring_hsize=None):
        super(RRNNforGRU, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.cell = RRNNforGRUCell(hidden_size, scoring_hsize=scoring_hsize)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, n_batch=None, device=torch.device('cpu')):
        if n_batch is None:
            return torch.zeros([self.batch_size, 1, self.hidden_size], requires_grad=True, device=device)
        else:
            return torch.zeros([n_batch, 1, self.hidden_size], requires_grad=True, device=device)
        
    def forward(self, inputs, G_structure_list=None):
        batch_size, time_steps, _ = inputs.shape
        h_next = self.init_hidden(batch_size, inputs.device)
        
        # stage 1
        if G_structure_list is None:
            h_list = []
            pred_tree_list = []
            margins_list = []
            structures_list = []
            pred_chars_list = []
    
            for t in range(time_steps):
                x_t = inputs[:, t, :].reshape(-1, 1, self.hidden_size)
                h_next, G_structure, G_node, G_margin = self.cell.forward_stage_search(x_t, h_next)
                
                pred_chars_list.append(self.output_layer(h_next))
                h_list.append(h_next)
                structures_list.append(G_structure)
                pred_tree_list.append(G_node)
                margins_list.append(G_margin)       
                        
            h_batch = torch.cat(h_list, dim=1)
            pred_chars_batch = torch.cat(pred_chars_list, dim=1)
            margins_batch = torch.stack(margins_list, dim=1)            
            
            return pred_chars_batch, h_batch, pred_tree_list, structures_list, margins_batch
        # stage 2
        else:
            pred_chars_list = []
            
            for t in range(time_steps):
                x_t = inputs[:, t, :].reshape(-1, 1, self.hidden_size)
                G_structure = G_structure_list[t]
                h_next = self.cell.foward_stage_fixing(x_t, h_next, G_structure)
                pred_chars_list.append(self.output_layer(h_next))
            
            pred_chars_batch = torch.cat(pred_chars_list, dim=1)

            return pred_chars_batch

if __name__ == '__main__':
    timer = time.time()
    HIDDEN_SIZE = 100
    batch_size = 16
    for i in range(1):
        x = torch.randn(batch_size, 1, HIDDEN_SIZE)
        h_prev = torch.randn(batch_size, 1, HIDDEN_SIZE)
        cell = RRNNforGRUCell(HIDDEN_SIZE)
        h_next, G_structure, G_node, G_margin = cell.forward_stage_search(x, h_prev)
    print(time.time()-timer)

    # model
    timer = time.time()
    for i in range(1):
        BATCH_SIZE = 16
        inputs = torch.randn(BATCH_SIZE, 20, HIDDEN_SIZE)
        model = RRNNforGRU(HIDDEN_SIZE, vocab_size=27, batch_size=BATCH_SIZE)
        pred_chars_batch, h_batch, pred_tree_list, structures_list, margins_batch = model.forward(inputs)
    print(time.time()-timer)

    GRU_structure = [['x', 'h_prev', 0, 'add', 'sigmoid'],
                     ['x', 'h_prev', 1, 'add', 'sigmoid'],
                     ['x', 'h_prev', 0, 'add', 'sigmoid'],
                     ['h_prev', 'G[1]', 2, 'mul', 'identity'],
                     ['x', 'G[3]', 2, 'add', 'tanh'],
                     ['o', 'G[0]', 2, 'add', 'minus'],
                     ['h_prev', 'G[2]', 3, 'mul', 'identity'],
                     ['G[4]', 'G[5]', 3, 'mul', 'identity'],
                     ['G[6]', 'G[7]', 3, 'add', 'identity']]

    # model
    timer = time.time()
    for i in range(1):
        BATCH_SIZE = 16
        inputs = torch.randn(BATCH_SIZE, 20, HIDDEN_SIZE)
        model = RRNNforGRU(HIDDEN_SIZE, vocab_size=27, batch_size=BATCH_SIZE)
        pred_chars_list = model.forward(inputs, [GRU_structure for _ in range(20)])
    print(time.time()-timer)
    
