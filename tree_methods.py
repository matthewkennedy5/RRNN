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

    o = torch.zeros(x.shape)
    r = torch.sigmoid(torch.mm(x, W_ir) + b_ir + torch.mm(h, W_hr) + b_hr)
    z = torch.sigmoid(torch.mm(x, W_iz) + b_iz + torch.mm(h, W_hz) + b_hz)
    rh = r*h
    h_tilde = torch.tanh(torch.mm(x, W_in) + b_in + torch.mm(rh, W_hn) + r*b_hn)
    z1 = 1-z
    zh = z*h
    zh_tilde = z1*h_tilde
    h_next = zh + zh_tilde

    rNode = Node(r, name='r', left_child=Node(x, 'x'), right_child=Node(h, 'h'))
    rhNode = Node(rh, name='r*h', left_child=rNode, right_child=Node(h, 'h'))
    h_tildeNode = Node(h_tilde, name='h_tilde', left_child=rhNode, right_child=Node(x, 'x'))
    z1Node = Node(z,  name='z1', left_child=Node(x, 'x'), right_child=Node(h, 'h'))
    z2Node = Node(z,  name='z2', left_child=Node(x, 'x'), right_child=Node(h, 'h'))
    zhNode = Node(zh, name='z*h', left_child=z1Node, right_child=Node(h, 'h'))
    oneMinuszNode = Node(z1, name='1-z', left_child=z2Node, right_child=Node(o, '0'))
    zh_tildeNode = Node(z1*h_tilde, name='(1-z)*h_tilde', left_child=h_tildeNode, right_child=oneMinuszNode)
    h_nextNode = Node(h_next, name='h_next', left_child=zh_tildeNode, right_child=zhNode)

    rNode.parent = rhNode
    rhNode.parent = h_tildeNode
    h_tildeNode.parent = zh_tildeNode
    z1Node.parent = zhNode
    z2Node.parent = oneMinuszNode
    zhNode.parent = h_nextNode
    oneMinuszNode.parent = zh_tildeNode
    zh_tildeNode.parent = h_nextNode
    rNode.leftchild.parent = rNode
    rNode.rightchild.parent = rNode
    rhNode.rightchild.parent = rhNode
    h_tildeNode.rightchild.parent = h_tildeNode
    z1Node.leftchild.parent = z1Node
    z1Node.rightchild.parent = z1Node
    z2Node.leftchild.parent = z2Node
    z2Node.rightchild.parent = z2Node
    zhNode.rightchild.parent = zhNode
    oneMinuszNode.rightchild.parent = oneMinuszNode

    node_list = [rNode, z1Node, z2Node, rhNode, h_tildeNode, zhNode, oneMinuszNode, zh_tildeNode, h_nextNode]
    return h_nextNode, node_list


#
#def GRUtree(x, h, Lr, Rr, Lz, Rz, Lh, Rh, br, bz, bh):
#    o = torch.zeros(x.shape)
#    r = torch.sigmoid(torch.mm(x, Lr) + torch.mm(h, Rr) + br)
#    z = torch.sigmoid(torch.mm(x, Lz) + torch.mm(h, Rz) + bz)
#    rh = r*h
#    z1 = 1-z
#    h_tilde = torch.tanh(torch.mm(x, Lh) + torch.mm(rh, Rh) + bh)
#    zh = z*h
#    zh_tilde = z1*h_tilde
#    h_next = zh+zh_tilde
#
#    rNode = Node(r, name='r', left_child=Node(x, 'x'), right_child=Node(h, 'h'))
#    rhNode = Node(rh, name='r*h', left_child=rNode, right_child=Node(h, 'h'))
#    h_tildeNode = Node(h_tilde, name='h_tilde', left_child=rhNode, right_child=Node(x, 'x'))
#    z1Node = Node(z,  name='z1', left_child=Node(x, 'x'), right_child=Node(h, 'h'))
#    z2Node = Node(z,  name='z2', left_child=Node(x, 'x'), right_child=Node(h, 'h'))
#    zhNode = Node(zh, name='z*h', left_child=z1Node, right_child=Node(h, 'h'))
#    oneMinuszNode = Node(z1, name='1-z', left_child=z2Node, right_child=Node(o, '0'))
#    zh_tildeNode = Node(z1*h_tilde, name='(1-z)*h_tilde', left_child=h_tildeNode, right_child=oneMinuszNode)
#    h_nextNode = Node(h_next, name='h_next', left_child=zh_tildeNode, right_child=zhNode)
#
#    rNode.parent = rhNode
#    rhNode.parent = h_tildeNode
#    h_tildeNode.parent = zh_tildeNode
#    z1Node.parent = zhNode
#    z2Node.parent = oneMinuszNode
#    zhNode.parent = h_nextNode
#    oneMinuszNode.parent = zh_tildeNode
#    zh_tildeNode.parent = h_nextNode
#    rNode.leftchild.parent = rNode
#    rNode.rightchild.parent = rNode
#    rhNode.rightchild.parent = rhNode
#    h_tildeNode.rightchild.parent = h_tildeNode
#    z1Node.leftchild.parent = z1Node
#    z1Node.rightchild.parent = z1Node
#    z2Node.leftchild.parent = z2Node
#    z2Node.rightchild.parent = z2Node
#    zhNode.rightchild.parent = zhNode
#    oneMinuszNode.rightchild.parent = oneMinuszNode
#
#    node_list = [rNode, z1Node, z2Node, rhNode, h_tildeNode, zhNode, oneMinuszNode, zh_tildeNode, h_nextNode]
#    return h_nextNode, node_list




def label(tree, l=1):
    tree.number = l
    if tree.leftchild is not None:
        label(tree.leftchild, 2*tree.number)
    if tree.rightchild is not None:
        label(tree.rightchild, 2*tree.number+1)
    return tree

def tree_matrixize(tree, mat):
    mat[tree.number-1, :] = tree.vector
    if tree.leftchild is not None:
        mat = tree_matrixize(tree.leftchild, mat)
    if tree.rightchild is not None:
        mat = tree_matrixize(tree.rightchild, mat)
    return mat

def tree_distance_metric(tree1, tree2):
    tree1 = label(tree1)
    tree2 = label(tree2)
    max_depth = max(depth(tree1), depth(tree2))
    mat1 = torch.zeros(2**max_depth-1, tree1.vector.shape[1])
    mat2 = torch.zeros(2**max_depth-1, tree1.vector.shape[1])
    mat1 = tree_matrixize(tree1, mat1)
    mat2 = tree_matrixize(tree2, mat2)
    return (mat1-mat2).norm()

def tree_distance_metric_list(pred_tree, target_tree, order=True, samples=10, device=torch.device('cpu')):
    # 不考虑target tree的不同order
    if order == False:
        for i in range(len(pred_tree)):
            tmp_list = []
            for j in range(len(target_tree)):
                tmp = tree_distance_metric(pred_tree[i], target_tree[j])
                tmp_list.append(tmp)
        return torch.tensor(min(tmp_list), device=device)
    # 考虑
    if order == True:
        # 随机选出 10 种构型
        res_list = []
        indicator_string_list = ['0'*len(target_tree)] + random.sample(["".join(seq) for seq in itertools.product("01", repeat=len(target_tree))], samples-1)
        for indicator_string in indicator_string_list:
            res = 0
            new_target_tree = target_tree.copy()
            for i in range(len(indicator_string)):  # 交换左右子树
                if indicator_string[i] == '1':
                    node = new_target_tree[i]
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
        return torch.tensor(min(res_list)/len(pred_tree), device=device)


