import os
import sys
import time
import datetime
import random
import torch
import torch.multiprocessing as mp
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# import tree_methods_parallel as tree_methods
import tree_methods
import dataloader
from GRU import RRNNforGRU
from structure_utils import structures_are_equal, GRU_STRUCTURE
import pickle
import pdb

VOCAB_SIZE = 27
HIDDEN_SIZE = 100
device = torch.device('cpu')
LOSS_FILE = 'loss.txt'
HYPERPARAM_FILE = 'hyperparameters.pkl'
RUNTIME_FILE = 'runtime.pkl'


class RRNNTrainer:
    """Trainer class for the RRNNforGRU.

    Inputs:
        model - RRNN model to train
        X_train - Training data. List of 3D torch tensors.
        y_train - Training labels (one-hot)
    """
    def __init__(self, model, gru_model, X_train, y_train, X_val, y_val, optimizer, params):
        self.model = model
        self.gru_model = gru_model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.optimizer = optimizer
        self.params = params
        self.lamb1, self.lamb2, self.lamb3, self.lamb4 = params['lambdas']
        # self.loss = torch.nn.KLDivLoss()
        self.loss = torch.nn.CrossEntropyLoss()
        self.iter_count = 0

    def batch_generator(self):
        epochs = self.params['epochs']
        batch_size = self.params['batch_size']
        for epoch in range(epochs):
            # Shuffle data
            shuffle_order = np.arange(len(self.X_train))
            np.random.shuffle(shuffle_order)
            self.X_train = self.X_train[shuffle_order]
            self.y_train = self.y_train[shuffle_order]

            if self.params['verbose']:
                print('\n\nEpoch ' + str(epoch + 1))

            # Checkpoint the model
            if epoch % self.params['epochs_per_checkpoint'] == 0:
                torch.save(self.model.state_dict(), 'epoch_%d.pt' % (epoch,))

            X_batches = []
            y_batches = []
            # process = 0
            n_processes = self.params['n_processes']
            partition_size = batch_size * n_processes
            for p in range(0, self.X_train.size()[0], partition_size):
                X_batches = []
                y_batches = []
                for b in range(p, p+partition_size, batch_size):
                    if b < len(self.X_train):
                        X_batch = self.X_train[b:b+batch_size]
                        y_batch = self.y_train[b:b+batch_size]
                        X_batches.append(X_batch)
                        y_batches.append(y_batch)
                yield X_batches, y_batches

    def train(self, epochs, n_processes=1):
        """Trains the RRNN for the given number of epochs.

        Inputs:
            epochs - Number of epochs (full passes over the training data) to train
                for

        Returns:
            loss_history - numpy array containing the loss at each training iteration.
            structure - The final structure of the RRNN tree.
        """
        N = len(self.X_train)
        iterations = epochs * N
        # set to training mode
        self.model.train()

        for X_batches, y_batches in self.batch_generator():
            processes = []
            for i in range(len(X_batches)):
                X_batch = X_batches[i]
                y_batch = y_batches[i]

                if self.params['debug']:
                    self.train_batch(X_batch, y_batch)
                else:
                    p = mp.Process(target=self.train_batch, args=(X_batch, y_batch))
                    p.start()
                    processes.append(p)
            for p in processes:
                p.join()

        self.model.eval()

    def train_batch(self, X_batch, y_batch):
         # zero gradient
        self.optimizer.zero_grad()

        batch_size = self.params['batch_size']
        loss_hist = np.zeros((batch_size, 4))
        loss_fn = 0
        for i in range(X_batch.size()[0]):
            x = X_batch[i]
            y = y_batch[i]
            loss = self.train_step(x, y)
            loss_fn += sum(loss)
            # TODO: Clean this up
            for l in range(4):
                try:
                    loss_hist[i, l] = loss[l].item()
                except AttributeError:
                    loss_hist[i, l] = loss[l]

        # Average out the loss
        loss_hist = np.mean(loss_hist, axis=0)
        loss_fn /= batch_size

        loss_fn.backward()
        self.optimizer.step()

        self.iter_count += 1

        # Save out the loss as we train because multiprocessing is weird with
        # instance variables
        train_loss = (loss_hist[0].item(), loss_hist[1].item(), loss_hist[2], loss_hist[3])
        with open(LOSS_FILE, 'a') as f:
            f.write('Train: %f %f %f %f' % train_loss)
            if len(self.X_val) > 0:
                val_loss = self.validation_loss()
                f.write('Val: %f %f %f %f' % val_loss)
        f.close()

        if self.params['verbose']:
            print('.', end='', flush=True)

    def validation_loss(self):
        """Runs inference over the validation set to calculate the validation loss.

        Returns:
            val_loss - tuple of loss values for each loss1, 2, 3, etc.
        """
        with torch.no_grad():
            n_val = len(self.X_val)
            val_loss = np.zeros(4)
            for i in range(n_val):
                x = self.X_val[i]
                y = self.y_val[i]
                val_loss += self.train_step(x, y)
            val_loss /= n_val
        return tuple(val_loss)

    def train_step(self, X, y):
        # TODO: Update this documentation to make it clear that it doesn't update weights
        """Performs a single iteration of training.

        Inputs:
            X - Embedded training sentence.
            y - True value for the final character that we're trying to predict.

        Returns:
            loss1 - Cross-entropy classification loss
            loss2 - SVM-like score margin loss
            loss3 - L2 regularization loss
            loss4 - Tree Distance Metric loss
        """
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
            loss1 = self.loss(out, torch.argmax(y).unsqueeze(0))
            # loss1 = self.loss(out, y.reshape(1,27).float())

        # loss2 is the negative sum of the scores (alpha) of the vector
        # corresponding to each node. It is an attempt to drive up the scores for
        # the correct vectors.
        loss2 = 0
        if self.lamb2 != 0:
            margin = self.params['loss2_margin']
            for s in range(len(scores)):
                difference = scores[s] - second_scores[s]
                if difference < margin:
                    # Here the subtraction comes from the fact that we want the
                    # loss to be 0 when the difference >= LOSS2_MARGIN,
                    # and equal to 1 when the difference is 0. Therefore,
                    # loss2 will always be between 0 and the number of
                    # vectors we have. We divide by LOSS2_MARGIN to scale
                    # the loss term to be between 0 and 1, so it LOSS2_MARGIN
                    # doesn't affect the overall scale of loss2.
                    value = torch.clamp(margin - difference, min=0) / margin
                    if value > 0:
                        loss2 += value

        loss3 = 0
        if self.lamb3 != 0:
            for param in self.model.parameters():
                loss3 += param.norm()

        loss4 = 0
        if self.lamb4 != 0:
            for l in range(len(pred_tree_list)):
                loss4 += tree_methods.tree_distance_metric_list(pred_tree_list[l],
                                                                target_tree_list[l],
                                                                samples=self.params['samples'],
                                                                device=device)

        # Record the structure
        structure_file = open('structure.txt', 'a')
        is_gru = structures_are_equal(structure, GRU_STRUCTURE)
        if is_gru:
            print('\nAcheived GRU structure!\n')
        structure_file.write(str(structure) + '\n\n')
        structure_file.close()

        return self.lamb1*loss1, self.lamb2*loss2, self.lamb3*loss3, self.lamb4*loss4

# Perform a training run using the given hyperparameters. Saves out data and model checkpoints
# into the current directory.
def run(params):
    # Assuming we are already in the directory where the output files should be
    pickle.dump(params, open(HYPERPARAM_FILE, 'wb'))
    print('[INFO] Saved hyperparameters.')

    start = time.time()
    gru_model = torch.load('../gru_parameters.pkl')
    model = RRNNforGRU(HIDDEN_SIZE, VOCAB_SIZE, params['multiplier'],
                       params['scoring_hidden_size'])

    model.share_memory()
    gru_model.share_memory()

    if params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    elif params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'])

    X_train, y_train, X_val, y_val = dataloader.load_normalized_data('../train20.txt',
                                                                     n_train=params['nb_data'],
                                                                     n_val=params['nb_val'],
                                                                     device=device,
                                                                     embeddings='gensim')

    trainer = RRNNTrainer(model, gru_model, X_train, y_train, X_val, y_val, optimizer, params)
    try:
        trainer.train(params['epochs'], n_processes=params['n_processes'])
    except ValueError:
        print('ValueError')
        gru_count = -1

    runtime = time.time() - start
    pickle.dump(runtime, open(RUNTIME_FILE, 'wb'))
    print('\n[INFO] Run complete.')

    torch.save(model.state_dict(), 'final_weights.pt')


if __name__ == '__main__':

    dirname = sys.argv[1]
    os.mkdir(dirname)
    os.chdir(dirname)

    params = {
        'learning_rate': 1e-5,
        'multiplier': 1e-3,
        'lambdas': (20, 1, 0, 2),
        'nb_data': 4998,
        'nb_val': 0,
        'epochs': 1,
        'n_processes': 1,
        'loss2_margin': 1,
        'scoring_hidden_size': 128,     # Set to None for no hidden layer
        'batch_size': 1,
        'verbose': True,
        'epochs_per_checkpoint': 100,
        'optimizer': 'adam',
        'samples': 10,
        'debug': True  # Turns multiprocessing off so pdb works
    }

    run(params)