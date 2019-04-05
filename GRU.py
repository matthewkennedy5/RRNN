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

from tree_methods import Node
import tree_methods


device = torch.device('cpu')
#if torch.cuda.is_available():
#    device = torch.device('cuda:0')

timer = time.time()

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
    # if vec.abs().max() > 1 and string not in ['tanh', 'sigmoid']:
    #     if vec.abs().max() > 1.05:
    #         print('\nVector element > 1.05: ' + string)
    #         print(vec[vec.abs() > 1])
    if string == 'tanh':
        return multiplier * torch.tanh(vec)
    elif string == 'sigmoid':
        return multiplier * torch.sigmoid(vec)
    elif string == 'minus':
        return 1-vec
    elif string == 'identity':
        return vec
    elif string == 'relu':
        return multiplier * torch.relu(vec)
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

    def __init__(self, hidden_size, multiplier, scoring_hsize=None):
        super(RRNNforGRUCell, self).__init__()
        self.multiplier = multiplier    # multiplier for the activation functions and initialization
                                        # of parameter matrices
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
            L = nn.Parameter(self.multiplier * weight_distribution.sample([hidden_size, hidden_size]))
            R = nn.Parameter(self.multiplier * weight_distribution.sample([hidden_size, hidden_size]))
            Ls.append(L)
            Rs.append(R)

        self.L_list = nn.ParameterList(Ls)
        self.R_list = nn.ParameterList(Rs)
        # The bias terms are initialized as zeros.
        self.b_list = nn.ParameterList([nn.Parameter(self.multiplier * torch.zeros(1, hidden_size)) for _ in range(self.l)])

        #################################################
        # Set L3, R3, b3 to the identity transformation #
        self.L_list[3] = nn.Parameter(torch.eye(hidden_size))
        self.R_list[3] = nn.Parameter(torch.eye(hidden_size))
        self.b_list[3] = nn.Parameter(torch.zeros(1, hidden_size))
        # Freeze them - they won't train to something else
        self.L_list[3].requires_grad = False
        self.R_list[3].requires_grad = False
        self.b_list[3].requires_grad = False
        #################################################

        if scoring_hsize is not None:
            self.scoring = nn.Sequential(
                                nn.Linear(hidden_size, scoring_hsize),
                                nn.ReLU(),
                                nn.Linear(scoring_hsize, 1)
                           )
        else:
            self.scoring = nn.Linear(hidden_size, 1, bias=False)

    def is_gru_pairing(self, r, left, right, l, binary_func, activation_func):
        """Returns True if the candidate vector is from the GRU truth tree.

        In the set of candidate vectors V_r, there exists one vector which
        corresponds to the GRU tree. This method returns true if the candidate
        vector defined by the inputs to this method is the GRU vector.

        Inputs:
            r - Which iteration of pairing we're on (0-8 for GRU)
            left - Left child node
            right - Right child node
            l - Choice of L, R, b weights (0-3 for GRU)
            binary_func - Binary function used ("add" or "mul")
            activation_func - Activation function used by the candidate vector
        """
        return False    # TODO: Implement

    def vector_is_in_gru(self, structure, r, candidate_names):
        # x = 0   # Indicies of x and h in the components_list array
        # h = 1
        gru_structure = {
            0: ['x', 'h', 0, 'add', 'sigmoid'],   # z1
            1: ['x', 'h', 1, 'add', 'sigmoid'],   # r
            2: ['x', 'h', 0, 'add', 'sigmoid'],   # z2
            3: ['G1', 'h', 3, 'mul', 'identity'], # r*h
            4: ['G3', 'x', 2, 'add', 'tanh'], # h_tilde
            5: ['G0', '0', 3, 'add', 'minus'],    # 1-z
            6: ['G4', 'G5', 3, 'mul', 'identity'],    # (1-z)*h_tilde
            7: ['G2', 'h', 3, 'mul', 'identity'], # z*h
            8: ['G6', 'G7', 3, 'add', 'identity']    # h_t}
        }
        true_structure = gru_structure[r]
        left, right, weight, binary, activation = range(5)
        # left and right are the same
        pred_left = candidate_names[structure[left]]
        pred_right = candidate_names[structure[right]]

        # This accounts for how the children may be swapped since the binary
        # functions are commutative.
        if (true_structure[left] != pred_left or true_structure[right] != pred_right) \
            and (true_structure[left] != pred_right or true_structure[right] != pred_left):
            return False
        if true_structure[binary] != structure[binary]:
            return False
        if true_structure[activation] != structure[activation]:
            return False
        return True

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
            candidate_names = ['x', 'h', '0'] + [comp.name for comp in components_list]
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

                                # Apply the activation function
                                # If the previous weights were the identity
                                # matrix (k == 3), then we're clear to use
                                # non-saturating activation functions.
                                if k != 3 and res.abs().max() >= 1:    # if the maximum entry of the vector is larger than 1, we could not use 1-x
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
                                    V_r.append(self.multiplier*torch.relu(res))
                                    V_structure.append([i, j, k, binary_func, 'relu'])

            # Prune the V_r to only include GRU vectors (should be 4 max)
            surviving_vectors = []
            surviving_structures = []
            for index, structure in enumerate(V_structure):
                # print(structure)
                if self.vector_is_in_gru(V_structure[index], r, candidate_names):
                    surviving_vectors.append(V_r[index])
                    surviving_structures.append(structure)

            V_r = surviving_vectors
            V_structure = surviving_structures

            scores_list = [self.scoring(v).item() for v in V_r] # calculate the scores for each vector
            max_index = np.argmax(scores_list)  # find the index of the vector with highest score
            # max_vector = V_r[max_index]
            max_structure = V_structure[max_index]

            TEMPERATURE = 1
            GUMBEL = False
            candidate_vectors = torch.stack(V_r)
            scores = self.scoring(candidate_vectors).squeeze()
            if GUMBEL:
                gumbel = -np.log(-np.log(np.random.uniform(0, 1, size=scores.shape)))
                gumbel = torch.tensor(gumbel).to(torch.float32)
                scores += gumbel
            probabilities = torch.nn.Softmax(dim=0)(scores / TEMPERATURE)

            if len(V_r) == 1:
                max_vector = V_r[0]
                # TODO: Deal with the margin when len(V_R) == 1
                second_vector = torch.zeros(max_vector.size())
                second_vectors.append(second_vector)
            else:
                max_vector = torch.zeros(V_r[0].size())
                for index, v in enumerate(V_r):
                    max_vector += v * probabilities[index]

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

        self.L_list[3].requires_grad = False
        self.R_list[3].requires_grad = False
        self.b_list[3].requires_grad = False

        return (h_next, G_forward, G_structure, components_list_forward, G_node,
                scores_list, second_scores_list)


class RRNNforGRU(nn.Module):

    def __init__(self, hidden_size, vocab_size, multiplier, scoring_hsize=None):
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





