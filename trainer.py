import time
import torch
import numpy as np

import tree_methods
from progressbar_utils import init_progress_bar
import dataloader
from GRU import RRNNforGRU
from structure_utils import structures_are_equal, GRU_STRUCTURE
import pickle
# 1106

VOCAB_SIZE = 27
HIDDEN_SIZE = 100
device = torch.device('cpu')


class RRNNTrainer:
    """Trainer class for the RRNNforGRU.

    Inputs:
        model - RRNN model to train
        X_train - Training data. List of 3D torch tensors.
        y_train - Training labels

    """
    def __init__(self, model, gru_model, X_train, y_train, optimizer, lambdas):
        self.model = model
        self.gru_model = gru_model
        self.X_train = X_train
        self.y_train = y_train
        self.optimizer = optimizer
        self.lamb1, self.lamb2, self.lamb3, self.lamb4 = lambdas
        self.loss = torch.nn.KLDivLoss()

    def train(self, epochs, verbose=True):
        """Trains the RRNN for the given number of epochs.

        Inputs:
            epochs - Number of epochs (full passes over the training data) to train
                for

        Returns:
            loss_history - numpy array containing the loss at each training iteration.
            structure - The final structure of the RRNN tree.
        """
        iterations = epochs * len(self.X_train)
        loss_history = np.zeros([iterations, 4])  # Columns are loss1, loss2, etc.
        gru_count = 0
        if verbose:
            bar = init_progress_bar(iterations)
            bar.start()
        for epoch in range(epochs):
            for i in range(len(self.X_train)):
                X = self.X_train[i]
                y = self.y_train[i]
                # set to training mode
                self.model.train()

                 # zero gradient
                self.optimizer.zero_grad()

                # forward pass and compute loss
                out, h_list, pred_tree_list, scores, second_scores, structure = self.model(X)

                # forward pass of traditional GRU
                gru_h_list = self.gru_model(X)[0]
                gru_h_list = torch.cat([torch.zeros(1,1, HIDDEN_SIZE), gru_h_list], dim=1)
                target_tree_list = []
                for t in range(X.shape[1]):
                    gru_x = X[:, t, :]
                    gru_h = gru_h_list[:, t, :]
                    target_tree = tree_methods.GRUtree_pytorch(gru_x, gru_h,
                                                               self.gru_model.weight_ih_l0,
                                                               self.gru_model.weight_hh_l0,
                                                               self.gru_model.bias_ih_l0,
                                                               self.gru_model.bias_hh_l0)[1]
                    target_tree_list.append(target_tree)

                # calculate loss function
                loss1 = 0
                if self.lamb1 != 0:
                    loss1 = self.loss(out, y.reshape(1,27).float())

                # loss2 is the negative sum of the scores (alpha) of the vector
                # corresponding to each node. It is an attempt to drive up the scores for
                # the correct vectors.
                loss2 = 0
                if self.lamb2 != 0:
                    loss2 = -np.sum(scores) + np.sum(second_scores)

                loss3 = 0
                if self.lamb3 != 0:
                    for param in self.model.parameters():
                        loss3 += param.norm()

                loss4 = 0
                if self.lamb4 != 0:
                    for l in range(len(pred_tree_list)):
                        loss4 += tree_methods.tree_distance_metric_list(pred_tree_list[l], target_tree_list[l], device=device)

                # compute gradient and take step in optimizer
                loss_fn = self.lamb1*loss1 + self.lamb2*loss2 + self.lamb3*loss3 + self.lamb4*loss4
                loss_fn.backward()
                self.optimizer.step()

                iteration = epoch * len(self.X_train) + i
                if structures_are_equal(structure, GRU_STRUCTURE):
                    print('Achieved the GRU structure on iteration', iteration)
                    gru_count += 1

                loss_history[iteration, :] = [self.lamb1*loss1, self.lamb2*loss2,
                                              self.lamb3*loss3, self.lamb4*loss4]
                if verbose:
                    bar.update(iteration + 1)

        if verbose:
            bar.finish()
        self.model.eval()
        return loss_history, gru_count


