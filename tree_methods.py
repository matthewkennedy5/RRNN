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

    o = torch.zeros(x.shape, device=x.device)
    r = torch.sigmoid(torch.matmul(x, W_ir) + b_ir + torch.matmul(h, W_hr) + b_hr)
    z = torch.sigmoid(torch.matmul(x, W_iz) + b_iz + torch.matmul(h, W_hz) + b_hz)
    rh = r*h
    h_tilde = torch.tanh(torch.matmul(x, W_in) + b_in + torch.matmul(rh, W_hn) + r*b_hn)
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



def label_dic(tree):
    l = 1
    d = {}
    def recursive(tree, l, d):
        if tree is not None:
            if (tree.leftchild is not None) or (tree.rightchild is not None):
                d[l] = tree.vector
                d = recursive(tree.leftchild, 2 * l, d)
                d = recursive(tree.rightchild, 2*l + 1, d)
        return d
    return recursive(tree, l, d)

    
def tree_distance_dic(tree1, tree2):
    d1 = label_dic(tree1)
    d2 = label_dic(tree2)
    s1 = set(d1.keys())   
    s2 = set(d2.keys())
    res = 0
    for key in s1 & s2:
        res += (d1[key]-d2[key]).norm()**2
    for key in s1 - s2:
        res += (d1[key]).norm()**2
    for key in s2 - s1:
        res += (d2[key]).norm()**2
    return res
        

def tree_distance_metric_list(pred_tree, target_tree):
    res = []
    for i in range(len(pred_tree)):
        vd_list = []
        tree1 = pred_tree[i]
        for j in range(len(target_tree)):
            tree2 = target_tree[j]
            vd = tree_distance_dic(tree1, tree2)
            vd_list.append(vd)
        res.append(torch.min(torch.stack(vd_list)))
    return sum(res)

