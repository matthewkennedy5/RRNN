# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 01:42:03 2018

@author: Bruce
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import time

from tree_methods import Node
import tree_methods


device = torch.device('cpu')
#if torch.cuda.is_available():
#    device = torch.device('cuda:0')

timer = time.time()
DEFAULT_MULTIPLIER = 1e-3

def retrieve_node(string, x, h_prev, G_node):
    '''
        return a Node instance with given vector
    '''
    if string == 'x':
        return Node(x, 'x')
    elif string == 'h':
        return Node(h_prev, 'h')
    elif string == '0':
        return Node(torch.zeros(x.shape, device=device), '0')
    elif string[0] == 'G':
        return G_node[int(string[1:])]
    else:
        raise ValueError

def retrieve_activation_func(string, vec, multiplier):
    '''
        return a vector with given activation function
    '''
    if vec.abs().max() > 1 and string not in ['tanh', 'sigmoid']:
        if vec.abs().max() > 1.05:
            print('\nVector element > 1.05: ')
            print(vec[vec.abs() > 1])
    if string == 'tanh':
        return multiplier * torch.tanh(vec)
    elif string == 'sigmoid':
        return multiplier * torch.sigmoid(vec)
    elif string == 'minus':
        return 1-vec
    elif string == 'identity':
        return vec
    else:
        raise ValueError

def retrieve_binary_func(string, vec1, vec2):
    '''
        return a vector that is calculated by given binary operation and two children vectors
    '''
    if string == 'add':
        return vec1 + vec2
    elif string == 'mul':
        return vec1 * vec2
    else:
        raise ValueError


class RRNNforGRUCell(nn.Module):

    def __init__(self, hidden_size, multiplier=DEFAULT_MULTIPLIER, scoring_hsize=None):
        super(RRNNforGRUCell, self).__init__()
        self.multiplier = multiplier    # multiplier for the activation functions and initialization
                                        # of parameter matrices
        self.hidden_size = hidden_size
        self.activations_func_list = ['sigmoid', 'tanh', 'minus', 'identity']
        self.binary_ops_list = ['mul', 'add']
        self.m = 1  # Num of output vectors
        self.N = 9  # Num of generated nodes in one cell
        self.l = 4  # Num of parameter matrices (L_i, R_i, b_i)
        self.L_list = nn.ParameterList([nn.Parameter(self.multiplier * torch.randn(hidden_size, hidden_size)) for _ in range(self.l)])
        self.R_list = nn.ParameterList([nn.Parameter(self.multiplier * torch.randn(hidden_size, hidden_size)) for _ in range(self.l)])
        self.b_list = nn.ParameterList([nn.Parameter(self.multiplier * torch.randn(1, hidden_size)) for _ in range(self.l)])

        if scoring_hsize is not None:
            self.scoring = nn.Sequential(
                                nn.Linear(hidden_size, scoring_hsize),
                                nn.ReLU(),
                                nn.Linear(scoring_hsize, 1)
                           )
        else:
            self.scoring = nn.Linear(hidden_size, 1, bias=False)

    def tree_structure_search(self, x, h_prev):
        """
        This is the "fake" forward process.
        Inputs:
            x: $x_t$ in the formulation of GRU, dimension: 1 * hidden_size
            h_prev: $h_{t-1}$ in the formulation of GRU, dimension: 1 * hidden_size
        Outputs:
            G:
            G_structure:
            components_list:
        """
        components_list = [] # explain on video chat
        G = []  # vectors of nodes in each cell
        G_structure = []    # corresponding structure
        # Gprime_structure = []
        second_vectors = []

        for r in range(self.N):
            candidate = [x, h_prev, torch.zeros(1, self.hidden_size, device=device)] + [comp.vector for comp in components_list]
            V_r = []    # same in algo 1 of doc, containing all possible vectors for next node
            V_structure = []    # contains the structure of each vector in V_r
            self.V_r = V_r
            self.V_structure = V_structure
            number_of_remaining_iters = self.N - r  # use this number to decide if you can only merge two existing components,
                                                    # not increase the number of components, or do whatever you want
            # starting the inner loop
            for i in range(len(candidate)-1):
                for j in range(i+1, len(candidate)):
                    # i will explain this part to you on video chat.
                    flag = 0
                    if number_of_remaining_iters == len(components_list) - 1: # You could only merge two components
                        if i >= 3:
                            flag = 1
                    elif number_of_remaining_iters == len(components_list): # could merge or keep # components to be the same
                        if j >= 3:
                            flag = 1
                    elif number_of_remaining_iters > len(components_list):  # whatever you want
                        flag = 1
                    else:
                        raise ValueError

                    if flag == 0:
                        pass
                    else:
                        s_i = candidate[i]
                        s_j = candidate[j]
                        for k in range(self.l):
                            L = self.L_list[k]
                            R = self.R_list[k]
                            b = self.b_list[k]
                            for binary_func in self.binary_ops_list:
                                if binary_func == 'add':
                                    res = torch.mm(s_i, L) + torch.mm(s_j, R) + b
                                else:   # elif binary_func == 'mul':
                                    res = torch.mm(s_i, L) * torch.mm(s_j, R) + b
                                if res.abs().max() >= 1:    # if the maximum entry of the vector is larger than 1, we could not use 1-x
                                                            # or x as the unary function, thus we can keep the entries within [-1, 1]
                                    V_r.append(self.multiplier*torch.sigmoid(res))
                                    V_structure.append([i, j, k, binary_func, 'sigmoid'])
                                    V_r.append(self.multiplier*torch.tanh(res))
                                    V_structure.append([i, j, k, binary_func, 'tanh'])
                                else:
                                    V_r.append(self.multiplier * torch.sigmoid(res))
                                    V_structure.append([i, j, k, binary_func, 'sigmoid'])
                                    V_r.append(self.multiplier * torch.tanh(res))
                                    V_structure.append([i, j, k, binary_func, 'tanh'])
                                    V_r.append(1-res)
                                    V_structure.append([i, j, k, binary_func, 'minus'])
                                    V_r.append(res)
                                    V_structure.append([i, j, k, binary_func, 'identity'])

            scores_list = [self.scoring(v).item() for v in V_r] # calculate the scores for each vector
            max_index = np.argmax(scores_list)  # find the index of the vector with highest score
            # max_vector = V_r[max_index]
            max_structure = V_structure[max_index]
            probabilities = torch.nn.Softmax(dim=0)(torch.tensor(scores_list))
            # if np.random.random() < 0.01:
            #     print(torch.max(probabilities).item())
            max_vector = torch.zeros(V_r[0].size())
            # START HERE: combine this code with the real forward pass.
            for index, v in enumerate(V_r):
                max_vector += v * probabilities[i]

            # Also grab the 2nd highest scoring vector and structure to be used
            # in the calculation of loss2
            second_index = np.where(np.argsort(scores_list) == 1)[0][0]
            second_vector = V_r[second_index]
            second_vectors.append(second_vector)

            # merge components
            i = max_structure[0]
            j = max_structure[1]
            name_list = ['x', 'h', '0']
            if i > 2 and j > 2: # if we combined two components
                left_node = components_list[i-3]
                right_node = components_list[j-3]
                new_component = Node(max_vector, 'G%d'%r, [left_node.name, right_node.name] + max_structure[2:])
                new_component.leftchild = left_node
                new_component.rightchild = right_node
                components_list[i-3] = new_component
                G.append(max_vector)
                G_structure.append([left_node.name, right_node.name] + max_structure[2:] + ['G%d'%r])
                del components_list[j-3]
            elif i <= 2 and j > 2: # if we only used one component, while another child is x, h, or 0.
                left_node = Node(candidate[i], name_list[i])
                right_node = components_list[j-3]
                new_component = Node(max_vector, 'G%d'%r, [left_node.name, right_node.name] + max_structure[2:])
                new_component.leftchild = left_node
                new_component.rightchild = right_node
                components_list[j-3] = new_component
                G.append(max_vector)
                G_structure.append([left_node.name, right_node.name] + max_structure[2:] + ['G%d'%r])
            else:   # elif i <= 2 and j <= 2:  # if both childs are from [x, h, 0]
                left_node = Node(candidate[i], name_list[i])
                right_node = Node(candidate[j], name_list[j])
                new_component = Node(max_vector, 'G%d'%r, [left_node.name, right_node.name] + max_structure[2:])
                new_component.leftchild = left_node
                new_component.rightchild = right_node
                components_list.append(new_component)
                G.append(max_vector)
                G_structure.append([left_node.name, right_node.name] + max_structure[2:] + ['G%d'%r])

        return G, G_structure, second_vectors, components_list

    def forward(self, x, h_prev):
        # G contains the vector values for each node
        G, G_structure, second_vectors, components_list = self.tree_structure_search(x, h_prev)
        G_node = [] # containing the Node class instance
        Gprime_node = []

        # Builds the tree from the bottom up
        for idx in range(len(G_structure)):
            left_name, right_name, k, binary_func, activation_func, name = G_structure[idx]

            L = self.L_list[k]
            R = self.R_list[k]
            b = self.b_list[k]
            left_node = retrieve_node(left_name, x, h_prev, G_node)
            right_node = retrieve_node(right_name, x, h_prev, G_node)
            if binary_func == 'add':
                res = torch.mm(left_node.vector, L) + torch.mm(right_node.vector, R) + b
            else:   # elif binary_func == 'mul':
                res = torch.mm(left_node.vector, L) * torch.mm(right_node.vector, R) + b

            res = retrieve_activation_func(activation_func, res, self.multiplier)

            node = Node(G[idx], name, structure=G_structure[idx], left_child=left_node, right_child=right_node)
            left_node.parent = node
            right_node.parent = node

            G_node.append(node)
            components_list_forward = [G_node[i] for i in range(len(G_node)) if G_node[i].parent is None]

        # G_forward = [node.vector for node in G_node]
        G_forward = [vector for vector in G]
        components_list_forward = None
        scores_list = [self.scoring(g) for g in G_forward]
        h_next = G_forward[-1]

        second_scores_list = [self.scoring(v) for v in second_vectors]
        return (h_next, G_forward, G_structure, components_list_forward, G_node,
                scores_list, second_scores_list)


class RRNNforGRU(nn.Module):

    def __init__(self, hidden_size, vocab_size, multiplier=DEFAULT_MULTIPLIER, scoring_hsize=None):
        super(RRNNforGRU, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.cell = RRNNforGRUCell(hidden_size, multiplier, scoring_hsize=scoring_hsize)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size, requires_grad=True, device=device)

    def forward(self, inputs):
        time_steps = inputs.size(1)
        h_next = self.init_hidden()
        h_list = []
        pred_tree_list = []
        scores = []
        structures = []
        pred_chars = []

        for t in range(time_steps):
            x_t = inputs[:, t, :].reshape(1, -1)
            h_next, G_forward, G_structure, components_list_forward, G_node, scores_list, second_scores_list = self.cell(x_t, h_next)
            pred_chars.append(self.output_layer(h_next))
            h_list.append(h_next)
            structures.append(G_structure)
            pred_tree_list.append(G_node)
            scores.append(scores_list)

        return (pred_chars, h_list, pred_tree_list, scores_list,
               second_scores_list, G_structure)


if __name__ == '__main__':

    # test case for RRNNforGRUCell
    x = torch.randn(1, 5)
    hidden = torch.randn(1, 5)
    model = RRNNforGRUCell(5)
    # TODO: Update to use torch.device rather than .cuda()
    if _cuda is True:
        x = x.cuda()
        hidden = hidden.cuda()
        model = model.cuda()

    G, G_structure, components_list = model.tree_structure_search(x, hidden)
    h_next, G_forward, G_structure, components_list_forward, G_node, scores_list = model(x, hidden)





