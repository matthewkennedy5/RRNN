# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 02:44:59 2018

@author: Bruce
"""

import torch
import itertools
import random

class Node(object):
    def __init__(self, vector, name=None, structure=None, left_child=None, right_child=None, parent=None):
        self.vector = vector
        self.name = name
        self.structure = structure
        self.leftchild = left_child
        self.rightchild = right_child
        self.parent = None

    def __str__(self):
        s = '''
            Name: %s
            structure: %s
            leftchild: %s
            rightchild: %s
            parent: %s
        '''%(self.name, str(self.structure), self.leftchild.name, self.rightchild.name, str(self.parent.name))
        return s

def depth(node):
    '''
        calculate the depth of a tree, which is defined by the maximum length of path that starts from the input node
    '''
    if node.leftchild == None and node.rightchild == None:
        return 1
    elif node.leftchild == None and node.rightchild != None:
        return depth(node.rightchild) + 1
    elif node.leftchild != None and node.rightchild == None:
        return depth(node.leftchild) + 1
    else:
        return max(depth(node.leftchild), depth(node.rightchild)) + 1

# GRU Tree
def GRUtree_pytorch(x, h, weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0):
    W_ir, W_iz, W_in = weight_ih_l0.chunk(3)
    W_hr, W_hz, W_hn = weight_hh_l0.chunk(3)
    b_ir, b_iz, b_in = bias_ih_l0.chunk(3)
    b_hr, b_hz, b_hn = bias_hh_l0.chunk(3)

    o = torch.zeros(x.shape)
    r = torch.sigmoid(torch.mm(x, W_ir) + b_ir + torch.mm(h, W_hr) + b_hr)
    z = torch.sigmoid(torch.mm(x, W_iz) + b_iz + torch.mm(h, W_hz) + b_hz)
    rh = r*h
    h_tilde = torch.tanh(torch.mm(x, W_in) + b_in + torch.mm(rh, W_hn) + r*b_hn)
    oneMinusz = 1-z
    zh = z*h
    zh_tilde = oneMinusz*h_tilde
    h_next = zh + zh_tilde

    z_1Node = Node(z,  name='z1', left_child=Node(x, 'x'), right_child=Node(h, 'h'))
    rNode = Node(r, name='r', left_child=Node(x, 'x'), right_child=Node(h, 'h'))
    z_2Node = Node(z,  name='z2', left_child=Node(x, 'x'), right_child=Node(h, 'h'))
    rhNode = Node(rh, name='r*h', left_child=Node(h, 'h'), right_child=rNode)
    h_tildeNode = Node(h_tilde, name='h_tilde', left_child=Node(x, 'x'), right_child=rhNode)
    oneMinuszNode = Node(oneMinusz, name='1-z', left_child=Node(o, '0'), right_child=z_1Node)
    zh_tildeNode = Node(zh_tilde, name='(1-z)*h_tilde', left_child=h_tildeNode, right_child=oneMinuszNode)
    zhNode = Node(zh, name='z*h', left_child=Node(h, 'h'), right_child=z_2Node)
    h_nextNode = Node(h_next, name='h_next', left_child=zh_tildeNode, right_child=zhNode)

    node_list = [z_1Node, rNode, z_2Node, rhNode, h_tildeNode, oneMinuszNode, zh_tildeNode, zhNode, h_nextNode]
    for node in node_list:
        node.leftchild.parent = node
        node.rightchild.parent = node

    return h_nextNode, node_list

def label(tree, l=1):
    """Assign index values to every node in a tree.
    As per the definition of the tree distance metric in the paper, the leaf
    nodes are not labelled. Trees are numbered in increasing order with the root
    at 1:
                        1
                       / \
                      2    3
                       \   /\
                        5 6  7
    Inputs:
        tree - Node instance containing the root of the tree
        l - Value to assign the root of the tree (default: 1).
    Returns:
        tree - The tree with all of the tree.number fields filled out with the
            correct indices.
    """
    if tree is not None:
        if tree.leftchild is not None or tree.rightchild is not None:
            tree.number = l
            label(tree.leftchild, 2 * l)
            label(tree.rightchild, 2*l + 1)
    return tree

# def label(tree, l=1):
#     tree.number = l
#     if tree.leftchild is not None:
#         label(tree.leftchild, 2*tree.number)
#     if tree.rightchild is not None:
#         label(tree.rightchild, 2*tree.number+1)
#     return tree

def tree_matrixize(tree, mat):
    if hasattr(tree, 'number'):
        mat[tree.number-1, :] = tree.vector
    if tree.leftchild is not None:
        mat = tree_matrixize(tree.leftchild, mat)
    if tree.rightchild is not None:
        mat = tree_matrixize(tree.rightchild, mat)
    return mat


# def tree_matrixize(tree, mat):
#     mat[tree.number-1, :] = tree.vector
#     if tree.leftchild is not None:
#         mat = tree_matrixize(tree.leftchild, mat)
#     if tree.rightchild is not None:
#         mat = tree_matrixize(tree.rightchild, mat)
#     return mat

def tree_distance_metric(tree1, tree2):
    tree1 = label(tree1)
    tree2 = label(tree2)
    max_depth = max(depth(tree1), depth(tree2))
    mat1 = torch.zeros(2**max_depth-1, tree1.vector.shape[1])
    mat2 = torch.zeros(2**max_depth-1, tree1.vector.shape[1])
    mat1 = tree_matrixize(tree1, mat1)
    mat2 = tree_matrixize(tree2, mat2)
    return (mat1-mat2).norm() ** 2

def tree_distance_metric_list(pred_tree, target_tree, order=True, samples=10, device=torch.device('cpu')):
    # if we don't consider the isomorphisms
    if order == False:
        for i in range(len(pred_tree)):
            tmp_list = []
            for j in range(len(target_tree)):
                tmp = tree_distance_metric(pred_tree[i], target_tree[j])
                tmp_list.append(tmp)
        return torch.tensor(min(tmp_list), device=device)
    # if we consider the isomorphsims
    if order == True:
        # randomly select 10 iso
        res_list = []
        # indicator_string_list = ['0'*len(target_tree)] + random.sample(["".join(seq) for seq in itertools.product("01", repeat=len(target_tree))], samples-1)
        indicator_string_list = ['000000000']   # Fixed target tree
        for indicator_string in indicator_string_list:
            res = 0
            new_target_tree = target_tree.copy()
            for i in range(len(indicator_string)):
                if indicator_string[i] == '1':
                    node = new_target_tree[i]
                    # Swap the left and right child
                    tmp = node.leftchild
                    node.leftchild = node.rightchild
                    node.rightchild = tmp
                    new_target_tree[i] = node
            for i in range(len(pred_tree)):
                tmp_list = []
                for j in range(len(new_target_tree)):
                    tmp = tree_distance_metric(pred_tree[i], new_target_tree[j])
                    tmp_list.append(tmp)
                res += min(tmp_list)
            res_list.append(res)
        # return torch.tensor(min(res_list)/len(pred_tree), device=device)

        # with open('TDM_losses.txt', 'a') as f:
        #     f.write('%f '*10 % tuple([loss.item() for loss in res_list]) + '\n')
        #     f.close()

        tree_distance = min(res_list) / len(pred_tree)
        return tree_distance
