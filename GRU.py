# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 01:42:03 2018

@author: Bruce
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import uniform
import numpy as np
import time
import pdb

from tree_methods import Node
import tree_methods


device = torch.device('cpu')
#if torch.cuda.is_available():
#    device = torch.device('cuda:0')

timer = time.time()


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
        # Freeze them - they won't train to something else
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
        sorted_indices = torch.sort(scores)[1]
        highest = sorted_indices[-1]
        second_highest = sorted_indices[-2]
        return scores[highest] - scores[second_highest]

    def forward(self, x, h_prev):
        # initialize L3 R3 to be identity
        l = self.l
        margins = []
        G_structure = [0, 1, 0]

        z_1 = torch.sigmoid(torch.mm(x, self.L_list[0]) + torch.mm(h_prev, self.R_list[0]) + self.b_list[0])

        r = torch.sigmoid(torch.mm(x, self.L_list[1]) + torch.mm(h_prev, self.R_list[1]) + self.b_list[1])

        z_2 = torch.sigmoid(torch.mm(x, self.L_list[0]) + torch.mm(h_prev, self.R_list[0]) + self.b_list[0])

        # h*r
        lst = torch.zeros(l, 1, self.hidden_size)
        for k in range(l):
            lst[k] = torch.mm(h_prev, self.L_list[k]) * torch.mm(r, self.R_list[k]) + self.b_list[k]
        scores = self.scoring(lst).squeeze()
        G_structure.append(torch.argmax(scores).item())
        rh = (lst*torch.nn.Softmax(dim=0)(scores).reshape(l, 1, 1)).sum(dim=0)
        margins.append(self.margin(scores))

        # h_tilde
        lst = torch.zeros(l, 1, self.hidden_size)
        for k in range(l):
            lst[k] = torch.tanh(torch.mm(x, self.L_list[k]) + torch.mm(rh, self.R_list[k]) + self.b_list[k])
        scores = self.scoring(lst).squeeze()
        G_structure.append(torch.argmax(scores).item())
        h_tilde = (lst*torch.nn.Softmax(dim=0)(scores).reshape(l, 1, 1)).sum(dim=0)
        margins.append(self.margin(scores))

        #oneMinusz
        lst = torch.zeros(l, 1, self.hidden_size)
        for k in range(l):
            lst[k] = 1 - (torch.mm(z_1, self.R_list[k]) + self.b_list[k])
        scores = self.scoring(lst).squeeze()
        G_structure.append(torch.argmax(scores).item())
        oneMinusZ1 = (lst*torch.nn.Softmax(dim=0)(scores).reshape(l, 1, 1)).sum(dim=0)
        margins.append(self.margin(scores))

        # (1-z)*h_tilde
        lst = torch.zeros(l, 1, self.hidden_size)
        for k in range(l):
            lst[k] = torch.mm(h_tilde, self.L_list[k])*torch.mm(oneMinusZ1, self.R_list[k]) + self.b_list[k]
        scores = self.scoring(lst).squeeze()
        G_structure.append(torch.argmax(scores).item())
        zh_tilde = (lst*torch.nn.Softmax(dim=0)(scores).reshape(l, 1, 1)).sum(dim=0)
        margins.append(self.margin(scores))

        # z2*h
        lst = torch.zeros(l, 1, self.hidden_size)
        for k in range(l):
            lst[k] = torch.mm(h_prev, self.L_list[k])*torch.mm(z_2, self.R_list[k]) + self.b_list[k]
        scores = self.scoring(lst).squeeze()
        G_structure.append(torch.argmax(scores).item())
        z2h = (lst*torch.nn.Softmax(dim=0)(scores).reshape(l, 1, 1)).sum(dim=0)
        margins.append(self.margin(scores))

        # h_t
        lst = torch.zeros(l, 1, self.hidden_size)
        for k in range(l):
            lst[k] = torch.mm(zh_tilde, self.L_list[k]) + torch.mm(z2h, self.R_list[k]) + self.b_list[k]
        scores = self.scoring(lst).squeeze()
        G_structure.append(torch.argmax(scores).item())
        h_next = (lst*torch.nn.Softmax(dim=0)(scores).reshape(l, 1, 1)).sum(dim=0)
        margins.append(self.margin(scores))

        o = torch.zeros(self.hidden_size)
        z_1Node = Node(z_1, name='z_1', structure=G_structure[0], left_child=Node(x, 'x'), right_child=Node(h_prev, 'h'))
        rNode = Node(r, name='r', structure=G_structure[1], left_child=Node(x, 'x'), right_child=Node(h_prev, 'h'))
        z_2Node = Node(z_1, name='z_1', structure=G_structure[2], left_child=Node(x, 'x'), right_child=Node(h_prev, 'h'))
        rhNode = Node(rh, name='r*h', structure=G_structure[3], left_child=Node(h_prev, 'h'), right_child=rNode)
        h_tildeNode = Node(h_tilde, name='h_tilde', structure=G_structure[4], left_child=Node(x, 'x'), right_child=rhNode)
        oneMinuszNode = Node(oneMinusZ1, name='1-z', structure=G_structure[5], left_child=Node(o, '0'), right_child=z_1Node)
        zh_tildeNode = Node(zh_tilde, name='(1-z)*h_tilde', structure=G_structure[6], left_child=h_tildeNode, right_child=oneMinuszNode)
        zhNode = Node(z2h, name='z*h', structure=G_structure[7], left_child=Node(h_prev, 'h'), right_child=z_2Node)
        h_nextNode = Node(h_next, name='h_next', structure=G_structure[8], left_child=zh_tildeNode, right_child=zhNode)

        G_node = [z_1Node, rNode, z_2Node, rhNode, h_tildeNode, oneMinuszNode, zh_tildeNode, zhNode, h_nextNode]
        for node in G_node:
            node.leftchild.parent = node
            node.rightchild.parent = node

        return h_next, G_structure, G_node, margins


class RRNNforGRU(nn.Module):

    def __init__(self, hidden_size, vocab_size, scoring_hsize=None):
        super(RRNNforGRU, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.cell = RRNNforGRUCell(hidden_size, scoring_hsize=scoring_hsize)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size, requires_grad=True, device=device)

    def forward(self, inputs):
        time_steps = inputs.size(1)
        h_next = self.init_hidden()
        h_list = []
        pred_tree_list = []
        margins = []
        structures = []
        pred_chars = []

        for t in range(time_steps):
            x_t = inputs[:, t, :].reshape(1, -1)
            h_next, G_structure, G_node, margins = self.cell(x_t, h_next)
            pred_chars.append(self.output_layer(h_next))
            h_list.append(h_next)
            structures.append(G_structure)
            pred_tree_list.append(G_node)

        return pred_chars, h_list, pred_tree_list, structures, margins


if __name__ == '__main__':

    # test case for RRNNforGRUCell
    x = torch.randn(1, 5)
    inputs = torch.randn(1, 3, 5)
    hidden = torch.randn(1, 5)
    model = RRNNforGRU(hidden_size=5, vocab_size=27, scoring_hsize=50)
    cell = model.cell

    (pred_chars, h_list, pred_tree_list, structures) = model(inputs)
    a = pred_tree_list[0]



