import os
import sys
import time
import datetime
import torch
import torch.multiprocessing as mp
import numpy as np

# import tree_methods_parallel as tree_methods
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
# TODO: Make this a hyperparameter
LOSS2_MARGIN = 2    # Beyond this value, differences in 1st vs. 2nd vector scores
                    # produce 0 loss.
LOSS_FILE = 'loss.pkl'
HYPERPARAM_FILE = 'hyperparameters.pkl'
RUNTIME_FILE = 'runtime.pkl'


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
        self.iter_count = 0

    def train(self, epochs, verbose=True, n_processes=1):
        """Trains the RRNN for the given number of epochs.

        Inputs:
            epochs - Number of epochs (full passes over the training data) to train
                for

        Returns:
            loss_history - numpy array containing the loss at each training iteration.
            structure - The final structure of the RRNN tree.
        """
        pickle.dump([], open(LOSS_FILE, 'wb'))
        iterations = epochs * len(self.X_train)
        self.gru_count = 0
        # set to training mode
        self.model.train()
        if verbose:
            self.bar = init_progress_bar(iterations)
            self.bar.start()
        for epoch in range(epochs):
            processes = []
            N = len(self.X_train)
            # TODO: Look up a better way to partition X_train.
            partition_size = N // n_processes + 1
            for partition in range(n_processes):
                start_index = partition * partition_size
                end_index = min(start_index + partition_size, N)
                p = mp.Process(target=self.train_partition,
                               args=(epoch, start_index, end_index, verbose))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

            # Checkpoint the model
            torch.save(self.model.state_dict(), 'epoch_%d.pt' % (epoch + 1,))
        if verbose:
            self.bar.finish()
        self.model.eval()

    def train_partition(self, epoch, start, end, verbose=False):
        """Performs one epoch of training on the given partition of the data."""
        X_partition = self.X_train[start:end]
        y_partition = self.y_train[start:end]
        for i in range(len(X_partition)):
            X = X_partition[i]
            y = y_partition[i]
            self.train_step(X, y)

            # if is_gru:
            #     print('Achieved the GRU structure on iteration', self.iter_count)
            #     self.gru_count += 1
            if verbose:
                self.bar.update(self.iter_count)

    def train_step(self, X, y):
        """Performs a single iteration of training.

        Inputs:
            X - Embedded training sentence.
            y - True value for the final character that we're trying to predict.

        Returns:
            loss - Numpy array containing [loss1, loss2, loss3, loss4] all
                multiplied by their respective lambdas
            is_gru - Boolean that is True if the structure of our RRNN
                is the GRU structure after the iteration.
        """
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
            for s in range(len(scores)):
                difference = scores[s] - second_scores[s]
                if difference < LOSS2_MARGIN:
                    # Here the subtraction comes from the fact that we want the
                    # loss to be 0 when the difference >= LOSS2_MARGIN,
                    # and equal to 1 when the difference is 0. Therefore,
                    # loss2 will always be between 0 and the number of
                    # vectors we have. We divide by LOSS2_MARGIN to scale
                    # the loss term to be between 0 and 1, so it LOSS2_MARGIN
                    # doesn't affect the overall scale of loss2.
                    loss2 += (LOSS2_MARGIN - difference) / LOSS2_MARGIN

        loss3 = 0
        if self.lamb3 != 0:
            for param in self.model.parameters():
                loss3 += param.norm()

        loss4 = 0
        if self.lamb4 != 0:
            for l in range(len(pred_tree_list)):
                loss4 += tree_methods.tree_distance_metric_list(pred_tree_list[l],
                                                                target_tree_list[l],
                                                                samples=5,
                                                                device=device)

        # compute gradient and take step in optimizer
        loss_fn = self.lamb1*loss1 + self.lamb2*loss2 + self.lamb3*loss3 + self.lamb4*loss4
        loss_fn.backward()
        self.optimizer.step()

        loss = np.array([self.lamb1*loss1, self.lamb2*loss2, self.lamb3*loss3, self.lamb4*loss4])
        is_gru = structures_are_equal(structure, GRU_STRUCTURE)
        self.iter_count += 1

        # Pickle out the loss as we train because multiprocessing is weird with
        # instance variables
        try:
            prev_loss = pickle.load(open(LOSS_FILE, 'rb'))
            prev_loss.append(loss)
            pickle.dump(prev_loss, open(LOSS_FILE, 'wb'))
        except Error:
            print('\nPickle Error')

        structure_file = open('structure.txt', 'a')
        structure_file.write(str(structure) + '\n\n')
        structure_file.close()
        # TODO: Figure out how to save GRU count / structure


# Perform a training run using the given hyperparameters. Saves out data and model checkpoints
# into the current directory.
def run(params):
    # Assuming we are already in the directory where the output files should be
    pickle.dump(params, open(HYPERPARAM_FILE, 'wb'))
    print('[INFO] Saved hyperparameters.')

    start = time.time()
    gru_model = torch.load('../gru_parameters.pkl')
    model = RRNNforGRU(HIDDEN_SIZE, VOCAB_SIZE, params['multiplier'])

    model.share_memory()
    gru_model.share_memory()

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    X_train, y_train = dataloader.load_normalized_data('../train20.txt',
                                                       embeddings='gensim')
    X_train = X_train[:params['nb_data']]
    y_train = y_train[:params['nb_data']]
    for i in range(len(X_train)):
        X_train[i] = X_train[i].to(device)
        y_train[i] = torch.tensor(y_train[i], device=device)

    trainer = RRNNTrainer(model, gru_model, X_train, y_train, optimizer,
                          params['lambdas'])
    try:
        trainer.train(params['epochs'], verbose=True,
                      n_processes=params['n_processes'])
    except ValueError:
        print('ValueError')
        gru_count = -1

    runtime = time.time() - start
    pickle.dump(runtime, open(RUNTIME_FILE, 'wb'))
    print('[INFO] Run complete.')

    torch.save(model.state_dict(), 'final_weights.pt')

if __name__ == '__main__':

    dirname = sys.argv[1]
    os.mkdir(dirname)
    os.chdir(dirname)

    params = {
        'learning_rate': 1e-5,
        'multiplier': 1e-3,
        'lambdas': (1000, 1, 0, 2e-1),
        'nb_data': 10,
        'epochs': 2,
        'n_processes': 5
    }

    run(params)
