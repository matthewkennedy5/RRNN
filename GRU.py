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

        if scoring_hsize is not None:
            self.scoring = nn.Sequential(
                                nn.Linear(hidden_size, scoring_hsize),
                                nn.ReLU(),
                                nn.Linear(scoring_hsize, 1)
                           )
        else:
            self.scoring = nn.Linear(hidden_size, 1, bias=False)


    def margin(self, scores):
        """Returns a Tensor of the difference between the top two scores.

        Input:
            scores - Tensor of output from the scoring net. Must contain at
                least two elements.
        """
        sorted_scores = torch.sort(scores)[0]
        return sorted_scores[:, :, -1] - sorted_scores[:, :, -2]


    def forward(self, x, h_prev):
        """ Batched version of forward pass

        Input:
            x - Tensor of shape batch_size * 1 * hidden_size
            h_prev - Tensor of shape batch_size * 1 * hidden_size
        """
        # initialize L3 R3 to be identity
        batch_size = x.shape[0]
        l = self.l
        margins = []
        G_structure = np.zeros((batch_size, self.N))
        softmax_func = torch.nn.Softmax(dim=2)

        z_1 = torch.sigmoid(torch.matmul(x, self.L_list[0]) + torch.matmul(h_prev, self.R_list[0]) + self.b_list[0])

        r = torch.sigmoid(torch.matmul(x, self.L_list[1]) + torch.matmul(h_prev, self.R_list[1]) + self.b_list[1])

        z_2 = torch.sigmoid(torch.matmul(x, self.L_list[0]) + torch.matmul(h_prev, self.R_list[0]) + self.b_list[0])

        # h*r
        lst = torch.zeros([batch_size, l, self.hidden_size], device=x.device)
        for k in range(l):
            lst[:, k, :] = (torch.matmul(h_prev, self.L_list[k]) * torch.matmul(r, self.R_list[k]) + self.b_list[k]).squeeze(dim=1)
        scores = self.scoring(lst).reshape(batch_size, 1, l)
        rh = torch.matmul(softmax_func(scores), lst)
        G_structure[:, 3] = torch.argmax(scores, dim=2).squeeze().tolist()
        margins.append(self.margin(scores))

        # h_tilde
        lst = torch.zeros([batch_size, l, self.hidden_size], device=x.device)
        for k in range(l):
            lst[:, k, :] = (torch.tanh(torch.matmul(x, self.L_list[k]) + torch.matmul(rh, self.R_list[k]) + self.b_list[k])).squeeze(dim=1)
        scores = self.scoring(lst).reshape(batch_size, 1, l)
        h_tilde = torch.matmul(softmax_func(scores), lst)
        G_structure[:, 4] = torch.argmax(scores, dim=2).squeeze().tolist()
        margins.append(self.margin(scores))

        #oneMinusz
        lst = torch.zeros([batch_size, l, self.hidden_size], device=x.device)
        for k in range(l):
            lst[:, k, :] = (1 - (torch.matmul(z_1, self.R_list[k]) + self.b_list[k])).squeeze(dim=1)
        scores = self.scoring(lst).reshape(batch_size, 1, l)
        oneMinusZ1 = torch.matmul(softmax_func(scores), lst)
        G_structure[:, 5] = torch.argmax(scores, dim=2).squeeze().tolist()
        margins.append(self.margin(scores))

        # (1-z)*h_tilde
        lst = torch.zeros([batch_size, l, self.hidden_size], device=x.device)
        for k in range(l):
            lst[:, k, :] = (torch.matmul(h_tilde, self.L_list[k])*torch.matmul(oneMinusZ1, self.R_list[k]) + self.b_list[k]).squeeze(dim=1)
        scores = self.scoring(lst).reshape(batch_size, 1, l)
        zh_tilde = torch.matmul(softmax_func(scores), lst)
        G_structure[:, 6] = torch.argmax(scores, dim=2).squeeze().tolist()
        margins.append(self.margin(scores))

        # z2*h
        lst = torch.zeros([batch_size, l, self.hidden_size], device=x.device)
        for k in range(l):
            lst[:, k, :] = (torch.matmul(h_prev, self.L_list[k])*torch.matmul(z_2, self.R_list[k]) + self.b_list[k]).squeeze(dim=1)
        scores = self.scoring(lst).reshape(batch_size, 1, l)
        z2h = torch.matmul(softmax_func(scores), lst)
        G_structure[:, 7] = torch.argmax(scores, dim=2).squeeze().tolist()
        margins.append(self.margin(scores))

        # h_t
        lst = torch.zeros([batch_size, l, self.hidden_size], device=x.device)
        for k in range(l):
            lst[:, k, :] = (torch.matmul(zh_tilde, self.L_list[k]) + torch.matmul(z2h, self.R_list[k]) + self.b_list[k]).squeeze(dim=1)
        scores = self.scoring(lst).reshape(batch_size, 1, l)
        h_next = torch.matmul(softmax_func(scores), lst)
        G_structure[:, 8] = torch.argmax(scores, dim=2).squeeze().tolist()
        margins.append(self.margin(scores))

#        # set up nodes
#        o = torch.zeros([batch_size, 1, self.hidden_size], device=x.device)
#        z_1Node = Node(z_1, name='z_1', structure=G_structure[0], left_child=Node(x, 'x'), right_child=Node(h_prev, 'h'))
#        rNode = Node(r, name='r', structure=G_structure[1], left_child=Node(x, 'x'), right_child=Node(h_prev, 'h'))
#        z_2Node = Node(z_2, name='z_2', structure=G_structure[2], left_child=Node(x, 'x'), right_child=Node(h_prev, 'h'))
#        rhNode = Node(rh, name='r*h', structure=G_structure[3], left_child=Node(h_prev, 'h'), right_child=rNode)
#        h_tildeNode = Node(h_tilde, name='h_tilde', structure=G_structure[4], left_child=Node(x, 'x'), right_child=rhNode)
#        oneMinuszNode = Node(oneMinusZ1, name='1-z', structure=G_structure[5], left_child=Node(o, '0'), right_child=z_1Node)
#        zh_tildeNode = Node(zh_tilde, name='(1-z)*h_tilde', structure=G_structure[6], left_child=h_tildeNode, right_child=oneMinuszNode)
#        zhNode = Node(z2h, name='z*h', structure=G_structure[7], left_child=Node(h_prev, 'h'), right_child=z_2Node)
#        h_nextNode = Node(h_next, name='h_next', structure=G_structure[8], left_child=zh_tildeNode, right_child=zhNode)

#        G_node = [z_1Node, rNode, z_2Node, rhNode, h_tildeNode, oneMinuszNode, zh_tildeNode, zhNode, h_nextNode]
#        for node in G_node:
#            node.leftchild.parent = node
#            node.rightchild.parent = node

        G_node = torch.cat([z_1, r, z_2, rh, h_tilde, oneMinusZ1, zh_tilde, z2h, h_next], dim=2)
        margins = torch.cat(margins, dim=1)
        # shape: batch_size * 1 * (nodes_num*hidden_size)

        return h_next, G_structure, G_node, margins


class RRNNforGRU(nn.Module):
    def __init__(self, hidden_size, vocab_size, batch_size, scoring_hsize=None):
        super(RRNNforGRU, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.cell = RRNNforGRUCell(hidden_size, scoring_hsize=scoring_hsize)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.batch_size = batch_size

    def init_hidden(self, n_batch, device=torch.device('cpu')):
        return torch.zeros([n_batch, 1, self.hidden_size], requires_grad=True, device=device)

    def forward(self, inputs):
        time_steps = inputs.shape[1]
        h_next = self.init_hidden(inputs.shape[0], device=inputs.device)
        h_list = []
        pred_tree_list = []
        margins_list = []
        structures_list = []
        pred_chars_list = []

        for t in range(time_steps):
            x_t = inputs[:, t, :].reshape(-1, 1, self.hidden_size)
            h_next, G_structure, G_node, margins = self.cell(x_t, h_next)

            pred_chars_list.append(self.output_layer(h_next))
            h_list.append(h_next)
            structures_list.append(G_structure)
            pred_tree_list.append(G_node)
            margins_list.append(margins)

        h_batch = torch.cat(h_list, dim=1)
        pred_chars_batch = torch.cat(pred_chars_list, dim=1)
        margins_batch = torch.stack(margins_list, dim=1)
        return pred_chars_batch, h_batch, pred_tree_list, structures_list, margins_batch

if __name__ == '__main__':
    timer = time.time()
    HIDDEN_SIZE = 100
    for i in range(1):
        x = torch.randn(16, 1, HIDDEN_SIZE)
        h_prev = torch.randn(16, 1, HIDDEN_SIZE)
        cell = RRNNforGRUCell(HIDDEN_SIZE)
        h_next, G_structure, G_node, margins = cell(x, h_prev)
    print(time.time()-timer)


    # model
    timer = time.time()
    for i in range(1):
        BATCH_SIZE = 16
        inputs = torch.randn(BATCH_SIZE, 20, HIDDEN_SIZE)
        model = RRNNforGRU(HIDDEN_SIZE, vocab_size=27, batch_size=BATCH_SIZE)
        pred_chars_batch, h_batch, pred_tree_list, structures_list, margins_batch = model(inputs)
    print(time.time()-timer)





