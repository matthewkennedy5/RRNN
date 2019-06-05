# -*- coding: utf-8 -*-
"""
Created on Mon May 27 19:00:13 2019

@author: Bruce
"""

import torch
import torch.nn as nn
from torch.distributions import uniform
import numpy as np


from tree_methods import Node

class BaseCell(nn.Module):
    def __init__(self, hidden_size, scoring_hsize=None):
        super(BaseCell, self).__init__()
        self.hidden_size = hidden_size
        self.binary_function_name_list = ['add', 'mul']
        self.binary_function_list = [lambda x,y: x+y, lambda x,y: x*y]
        self.unary_function_name_list = ['sigmoid', 'tanh', 'oneMinus', 'identity', 'relu']
        self.unary_function_list = [torch.sigmoid, torch.tanh, lambda x: 1-x, lambda x: x, torch.relu]
        self.m = 1  # Num of output vectors
        self.N = 9  # Num of generated nodes in one cell
        self.l = 4  # Num of parameter matrices (L_i, R_i, b_i)

        # Initalize L R weights from the uniform distribution and b to be zero
        weight_distribution = uniform.Uniform(-1/np.sqrt(hidden_size), 1/np.sqrt(hidden_size))
        self.L_list = nn.ParameterList([nn.Parameter(weight_distribution.sample([hidden_size, hidden_size])) for _ in range(self.l)])
        self.R_list = nn.ParameterList([nn.Parameter(weight_distribution.sample([hidden_size, hidden_size])) for _ in range(self.l)])
        self.b_list = nn.ParameterList([nn.Parameter(torch.zeros(1, hidden_size)) for _ in range(self.l)])

        # Set L3, R3, b3 to the identity transformation #
        self.L_list[3] = nn.Parameter(torch.eye(hidden_size))
        self.R_list[3] = nn.Parameter(torch.eye(hidden_size))
        self.b_list[3] = nn.Parameter(torch.zeros(1, hidden_size))
        
        # TODO: more complicated scoring
        if scoring_hsize is not None:
            self.scoring = nn.Sequential(
                                nn.Linear(hidden_size, scoring_hsize),
                                nn.ReLU(),
                                nn.Linear(scoring_hsize, 1))
        else:
            self.scoring = nn.Linear(hidden_size, 1, bias=False)
        
        # softmax function
        self.softmax_func = torch.nn.Softmax(dim=2)
        
    def margin(self, scores):
        """Returns a Tensor of the difference between the top two scores.
        Input:
            scores - Tensor of output from the scoring net. Shape: 1 * ?
        """
        sorted_scores = torch.sort(scores)[0]
        return sorted_scores[0, -1] - sorted_scores[0, -2]
    
    
class RRNNGRUCell(BaseCell):
    def forward(self, x, h):
        batch_size = x.shape[0]
        G_structure = [0,1,0]
        margins = []
        
        z_1 = torch.sigmoid(torch.matmul(x, self.L_list[0]) + torch.matmul(h, self.R_list[0]) + self.b_list[0])
        r = torch.sigmoid(torch.matmul(x, self.L_list[1]) + torch.matmul(h, self.R_list[1]) + self.b_list[1])
        z_2 = torch.sigmoid(torch.matmul(x, self.L_list[0]) + torch.matmul(h, self.R_list[0]) + self.b_list[0])        
        
        # h*r
        lst = []
        for k in range(self.l):
            lst.append((torch.matmul(h, self.L_list[k]) + torch.matmul(r, self.R_list[k]) + self.b_list[k]))
        lst = torch.cat(lst, dim=1)
        scores = self.scoring(lst).reshape(batch_size, 1, self.l)
        rh = torch.matmul(self.softmax_func(scores), lst)
        scores_sum_batch = scores.sum(dim=0)
        G_structure.append(torch.argmax(scores_sum_batch.sum(dim=0)).item())
        margins.append(self.margin(scores_sum_batch))
        
        # h_tilde = tanh(Wh*x + Wh'*rh + bh)
        lst = []
        for k in range(self.l):
            lst.append((torch.matmul(x, self.L_list[k]) + torch.matmul(rh, self.R_list[k]) + self.b_list[k]))
        lst = torch.cat(lst, dim=1)
        scores = self.scoring(lst).reshape(batch_size, 1, self.l)
        h_tilde = torch.matmul(self.softmax_func(scores), lst)
        scores_sum_batch = scores.sum(dim=0)
        G_structure.append(torch.argmax(scores_sum_batch.sum(dim=0)).item())
        margins.append(self.margin(scores_sum_batch))
        
        # oneMinusz1 = 1-z1
        lst = []
        for k in range(self.l):
            lst.append(1 - (torch.matmul(z_1, self.R_list[k]) + self.b_list[k]))
        lst = torch.cat(lst, dim=1)
        scores = self.scoring(lst).reshape(batch_size, 1, self.l)
        oneMinusZ1 = torch.matmul(self.softmax_func(scores), lst)
        scores_sum_batch = scores.sum(dim=0)
        G_structure.append(torch.argmax(scores_sum_batch.sum(dim=0)).item())
        margins.append(self.margin(scores_sum_batch))
        
        # zh_tilde = (1-z1)*h_tilde
        lst = []
        for k in range(self.l):
            lst.append(torch.matmul(h_tilde, self.L_list[k])*torch.matmul(oneMinusZ1, self.R_list[k]) + self.b_list[k])
        lst = torch.cat(lst, dim=1)
        scores = self.scoring(lst).reshape(batch_size, 1, self.l)
        zh_tilde = torch.matmul(self.softmax_func(scores), lst)
        scores_sum_batch = scores.sum(dim=0)
        G_structure.append(torch.argmax(scores_sum_batch.sum(dim=0)).item())
        margins.append(self.margin(scores_sum_batch))
                
        # z2*h
        lst = []
        for k in range(self.l):
            lst.append(torch.matmul(h, self.L_list[k])*torch.matmul(z_2, self.R_list[k]) + self.b_list[k])
        lst = torch.cat(lst, dim=1)
        scores = self.scoring(lst).reshape(batch_size, 1, self.l)
        z2h = torch.matmul(self.softmax_func(scores), lst)
        scores_sum_batch = scores.sum(dim=0)
        G_structure.append(torch.argmax(scores_sum_batch.sum(dim=0)).item())
        margins.append(self.margin(scores_sum_batch))
                
        # h_t = zh_tilde + z2h
        lst = []
        for k in range(self.l):
            lst.append(torch.matmul(zh_tilde, self.L_list[k]) + torch.matmul(z2h, self.R_list[k]) + self.b_list[k])
        lst = torch.cat(lst, dim=1)
        scores = self.scoring(lst).reshape(batch_size, 1, self.l)
        h_next = torch.matmul(self.softmax_func(scores), lst)
        scores_sum_batch = scores.sum(dim=0)
        G_structure.append(torch.argmax(scores_sum_batch.sum(dim=0)).item())
        margins.append(self.margin(scores_sum_batch))

        #
        G_node = torch.cat([z_1, r, z_2, rh, h_tilde, oneMinusZ1, zh_tilde, z2h, h_next], dim=1)
        margins = torch.stack(margins)
        
        return h_next, G_structure, G_node, margins

class RRNNCell(BaseCell):
    def forward(self, x, h):
        
        pass

class RRNNmodel(nn.Module):
    def __init__(self, batch_size, num_time_step, hidden_size, vocab_size, cell_strucutre='RRNNGRU', scoring_hsize=None):
        super(RRNNmodel, self).__init__()
        self.batch_size = batch_size
        self.num_time_step = num_time_step
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        
        self.cell_structure = cell_strucutre
        if cell_strucutre == 'RRNNGRU':
            self.cell = RRNNGRUCell(hidden_size, scoring_hsize)
        elif cell_strucutre == 'RRNN':
            pass
        else:
            raise ValueError('Unsupported Cell structure: %s'%(str(cell_strucutre)))

    def init_hidden(self, n_batch=None, device=torch.device('cpu')):
        if n_batch is None:
            return torch.zeros([self.batch_size, 1, self.hidden_size], requires_grad=True, device=device)
        else:
            return torch.zeros([n_batch, 1, self.hidden_size], requires_grad=True, device=device)
        
    def forward(self, inputs):
        """
            inputs should be shape of batch_size * num_time_step * hidden_size
        """
        assert list(inputs.shape) == [self.batch_size, self.num_time_step, self.hidden_size]
        h_next = self.init_hidden(inputs.shape[0], inputs.device)
        
        h_list = []
        pred_chars_list = []
        structures_list = []
        pred_tree_list = []
        margins_list = []
        
        for t in range(self.num_time_step):
            x = inputs[:, t, :].reshape(-1, 1, self.hidden_size)
            h_next, G_structure, G_node, G_margin = self.cell(x, h_next)
            
            h_list.append(h_next)
            pred_chars_list.append(self.output_layer(h_next))
            structures_list.append(G_structure)
            pred_tree_list.append(G_node)
            margins_list.append(G_margin) 
        
        return h_list, pred_chars_list, structures_list, pred_tree_list, margins_list




if __name__ == '__main__':
    BATCHSIZE = 64
    HIDDENSIZE = 100
    x = torch.randn(BATCHSIZE, 1, HIDDENSIZE)
    h = torch.zeros(BATCHSIZE, 1, HIDDENSIZE)
    cell = RRNNGRUCell(HIDDENSIZE, 128)
    h_next, G_structure, G_node, margins = cell(x, h)
    
    inputs = torch.randn(BATCHSIZE, 20, HIDDENSIZE)
    model = RRNNmodel(BATCHSIZE, 20, HIDDENSIZE, 27)
    h_list, pred_chars_list, structures_list, pred_tree_list, margins_list = model(inputs)
    
    

    