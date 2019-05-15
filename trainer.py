import os, sys, platform
import time
import datetime
import torch
from torch import nn
import torchtext
import numpy as np
from torch.utils.data import DataLoader
import time
import json
from pprint import pprint

import tree_methods
from GRU import RRNNforGRU
import pickle
import pdb
from tqdm import tqdm, trange
import standard_data

VOCAB_SIZE = 27
HIDDEN_SIZE = 100
TRAIN_LOSS_FILE = 'loss.txt'
TRAIN_ACC_FILE = 'train_acc.txt'
VAL_LOSS_FILE = 'val_loss.txt'
VAL_ACC_FILE = 'val_acc.txt'
HYPERPARAM_FILE = 'hyperparameters.pkl'
RUNTIME_FILE = 'runtime.pkl'
N_LOSS_TERMS = 4


class RRNNTrainer:
    """Trainer class for the RRNNforGRU.

    Inputs:
        model - RRNN model to train
        X_train - Training data. List of 3D torch tensors.
        y_train - Training labels (one-hot)
    """
    def __init__(self, model, gru_model, train_dataloader, val_dataloader, params):
        self.model = model
        self.gru_model = gru_model
        self.train_data = train_dataloader
        self.val_data = val_dataloader

        self.train_mode = params['initial_train_mode']
        scoring, weights = self.get_parameter_subsets()
        learning_rate = params['learning_rate']
        if params['optimizer'] == 'adam':
            self.scoring_optimizer = torch.optim.Adam(scoring, lr=learning_rate)
            self.weights_optimizer = torch.optim.Adam(weights, lr=learning_rate)
        elif params['optimizer'] == 'sgd':
            self.scoring_optimizer = torch.optim.SGD(scoring, lr=learning_rate)
            self.weights_optimizer = torch.optim.SGD(weights, lr=learning_rate)
        if self.train_mode == 'scoring':
            self.optimizer = self.scoring_optimizer
        else:
            self.optimizer = self.weights_optimizer

        self.params = params
        self.lamb1, self.lamb2, self.lamb3, self.lamb4 = params['lambdas']
        # self.loss = torch.nn.KLDivLoss()
        self.loss = torch.nn.CrossEntropyLoss()

    def get_parameter_subsets(self):
        """Separates the parameters into the scoring and weights.

        The output layer is included in both scoring and weights since we always
        train it.

        Returns:
            scoring - list of the scoring parameters (nn.Parameter)
            weights - list of the L R b weight parameters
        """
        scoring = []
        weights = []
        for name, param in self.model.named_parameters():
            if 'output' in name:
                scoring.append(param)
                weights.append(param)
            elif 'scoring' in name:
                scoring.append(param)
            elif '_list' in name:   # L_list R_list and b_list
                weights.append(param)
            else:
                raise ValueError('Unknown parameter name: ' + name)
        return scoring, weights

    def switch_train_mode(self):
        """Switches the train mode and freezes parameters from the other mode.

        The training mode corresponds to which parameters we're trying to train.
        We are alternating between training the scoring NN and the L, R, b,
        output weights of the RRNN, since training them both at the same time
        interferes with each other.
        """
        print('[INFO] Switching to training the ' + self.train_mode + '.')
        if self.train_mode == 'weights':
            self.train_mode = 'scoring'
            self.optimizer = self.scoring_optimizer
        else:
            self.train_mode = 'weights'
            self.optimizer = self.weights_optimizer
        # self.freeze_params()

    def freeze_params(self):
        """Freeze the parameters we're not trying to train.

        Depending on the epoch, it will either freeze L R b and train scoring,
        or freeze scoring and train the L R b weights. The output layer is never
        frozen.
        """
        names = [name for name, _ in self.model.named_parameters()]
        freeze = []
        if self.train_mode == 'scoring':
            for name in names:
                if ('L_list' in name or 'R_list' in name or 'b_list' in name):
                    freeze.append(name)
        elif self.train_mode =='weights':
            for name in names:
                if 'scoring' in name:
                    freeze.append(name)
        for name, param in self.model.named_parameters():
            if name in freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def checkpoint_model(self, epoch):
        save_name = 'checkpoint_' + str(epoch) + '_' + str(time.time()) + '.pt'
        torch.save(self.model.state_dict(), save_name)
        print('[INFO] Checkpointed the model.')
        torch.save(self.weights_optimizer.state_dict(), 'weights_optimizer.pt')
        torch.save(self.scoring_optimizer.state_dict(), 'scoring_optimizer.pt')
        print('[INFO] Saved the optimizer state.')

    def train(self, n_epochs):
        """Trains the RRNN for the given number of epochs.

        Inputs:
            n_epochs - Number of full passes over the training set

        Returns:
            loss_history - Tensor of shape (iterations, loss terms) giving the training
                loss1 ... loss4 through each iteration.
            acc_history - Tensor of shape (iterations, 1) giving the training accuracy
                for each training iteration.
            structure_history - Tensor of shape (iterations, time_steps, structure_len)
                containing the structure of each tree for each iteration (includes
                all timesteps).
        """
        self.model.train()
        self.gru_model.train()

        loss_history = []
        acc_history = []
        structure_history = []
        start_time = time.time()
        for e in range(n_epochs):
            print('\n[INFO] Epoch %d/%d' % (e+1, n_epochs))
            with tqdm(self.train_data) as t:
                for X_batch, y_batch in self.train_data:
                    self.optimizer.zero_grad()
                    losses, acc, structures = self.train_step_cuda(X_batch, y_batch)
                    loss_fn = sum(losses)
                    loss_fn.backward()
                    self.optimizer.step()

                    losses = printable(losses)
                    loss_history.append(losses)
                    acc_history.append(acc)
                    # structure_history.append(structures)  # This file gets huge

                    t.set_postfix(loss=losses)
                    t.update()

            if e % self.params['validate_every'] == 0:
                if len(self.val_data) > 0:
                    self.validate()

            if e % self.params['epochs_per_checkpoint'] == 0:
                self.checkpoint_model(e+1)

            if e % self.params['pickle_every'] == 0:
                pickle.dump(loss_history, open('loss_history.pkl', 'wb'))
                pickle.dump(acc_history, open('acc_history.pkl', 'wb'))
                pickle.dump(structure_history, open('structure_history.pkl', 'wb'))
                print('[INFO] Saved loss, accuracy, and structure history.')

            if e % self.params['alternate_every'] == 0:
                self.switch_train_mode()

            if time.time() - start_time > self.params['max_time']:
                print('[INFO] Maximum time exceeded:', time.time() - start_time)
                break

        self.model.eval()
        self.gru_model.eval()
        return loss_history, acc_history, structure_history

    def train_step_cuda(self, X, y):
        """Calculates loss and accuracy for the given batch of data points.

        Inputs:
            X: Tensor of shape (batch_size, time_steps, hidden_size)
            y: Tensor of shape (batch_size, time_steps, hidden_size), one-hot

        Returns:
            losses
            accuracy

        Requires:
            self.model = RRNNforGRU
            self.gru_model
            self.params
            self.loss = torch.nn.CrossEntropyLoss()
        """
        batch_size, time_steps, hidden_size = X.shape

        # forward pass
        pred_chars_batch, h_batch, pred_tree_list, structures_list, margins_batch = self.model(X)

        # forward pass of ground-truth GRU
        gru_h_list = self.gru_model(X)[0]
        gru_h_list = torch.cat([torch.zeros([batch_size, 1, HIDDEN_SIZE], device=X.device), gru_h_list], dim=1)
        target_tree_list = []
        for t in range(time_steps):
            gru_x = X[:, t, :].reshape(batch_size, 1, HIDDEN_SIZE)
            gru_h = gru_h_list[:, t, :].reshape(batch_size, 1, HIDDEN_SIZE)
            target_tree = tree_methods.GRUtree_pytorch(gru_x, gru_h,
                                                       self.gru_model.weight_ih_l0,
                                                       self.gru_model.weight_hh_l0,
                                                       self.gru_model.bias_ih_l0,
                                                       self.gru_model.bias_hh_l0)[1]
            target_tree_list.append(target_tree)

        # get weight for each loss terms
        lamb1, lamb2, lamb3, lamb4 = self.params['lambdas']

        # calculate loss terms
        loss1 = 0
        if lamb1 != 0:
            if y.dim() == 1:    # There is one output at the end of the sequence (sst)
                loss1 = self.loss(pred_chars_batch[:, -1, :], y)
            else:   # There is output for every time step
                for i_time in range(time_steps):
                    loss1 += self.loss(pred_chars_batch[:, i_time, :], y[:, i_time])

        loss2 = 0
        if lamb2 != 0:
            desired_margin = self.params['loss2_margin']
            loss2 = (desired_margin - margins_batch.clamp(max=desired_margin)).sum().div_(desired_margin)
            loss2 /= batch_size

        loss3 = 0
        if lamb3 != 0:
            for param in self.model.parameters():
                loss3 += param.norm()**2

        loss4 = 0
        if lamb4 != 0:
            pred_tree_list = torch.cat(pred_tree_list, dim=1)
            target_tree_list = torch.cat(target_tree_list, dim=1)
            loss4 = (pred_tree_list-target_tree_list).norm()**2
            loss4 /= batch_size

        losses = [lamb1*loss1, lamb2*loss2, lamb3*loss3, lamb4*loss4]
        if y.dim() == 1:    # SST dataset
            accuracy = (pred_chars_batch.argmax(dim=2)[:, -1] == y).sum().item() / float(batch_size)
        else:
            accuracy = (pred_chars_batch.argmax(dim=2) == y).sum().item() / float(time_steps * batch_size)

        return losses, accuracy, structures_list

    def validate(self, verbose=True):
        """Runs inference over the validation set periodically during training.

        Prints the validation loss and accuracy to their respective files.
        """
        X_val, y_val = next(iter(self.val_data))
        losses, accuracy, _ = self.train_step_cuda(X_val, y_val)

        print('val_loss:', printable(losses), 'val_acc:', accuracy)
        with open(VAL_LOSS_FILE, 'a') as f:
            f.write('%f %f %f %f\n' % tuple(losses))
        f.close()
        with open(VAL_ACC_FILE, 'a') as f:
            f.write('%f\n' % (accuracy,))
        f.close()


def printable(x):
    """Converts a tuple or list containing tensors and numbers to just numbers.

    Also rounds the numbers to 4 decimal places.

    Input:
        input - A list or tuple containing tensors and/or numbers.

    Returns:
        result - A tuple with .item() called on the tensors.
    """
    result = list(x)
    for i, l in enumerate(x):
        try:
            result[i] = l.item()
        except AttributeError:
            result[i] = l
        result[i] = round(result[i], 4)

    return result


# Performs a training run using the given hyperparameters. Saves out data and model checkpoints
# into the current directory.
def run(params):
    # Assuming we are already in the run output directory
    pickle.dump(params, open(HYPERPARAM_FILE, 'wb'))
    print('[INFO] Saved hyperparameters.')

    start = time.time()
    device = torch.device(params['device'])
    gru_model = torch.load('../gru_parameters_new.pt').to(device)

    if params['warm_start']:
        weights = params['weights_file']
        print('[INFO] Warm starting from ' + weights + '.')
        model.load_state_dict(torch.load(weights))

    if params['dataset'] == 'wiki':
        print('[INFO] Loading the enwik8 dataset.')
        train_set = standard_data.EnWik8Clean(subset='train', n_data=params['nb_train'], device=device)
        validation_set = standard_data.EnWik8Clean(subset='val', n_data=params['nb_val'], device=device)
        num_classes = 27
    elif params['dataset'] == 'ptb':
        print('[INFO] Loading the Penn Treebank dataset.')
        dirname = os.getcwd()
        os.chdir('..')
        train_set = standard_data.PennTreebank(subset='train', n_data=params['nb_train'], device=device)
        validation_set = standard_data.PennTreebank(subset='val', n_data=params['nb_val'], device=device)
        os.chdir(dirname)
        num_classes = 10001
    elif params['dataset'] == 'sst':
        print('[INFO] Loading the SST dataset.')
        dirname = os.getcwd()
        os.chdir('..')
        train_set = standard_data.SST(subset='train', n_data=params['nb_train'], device=device)
        validation_set = standard_data.SST(subset='val', n_data=params['nb_val'], device=device)
        os.chdir(dirname)
        num_classes = 3
    else:
        raise ValueError('Invalid dataset name. The "dataset" field in the JSON parameter file'
                         ' must be "wiki", "ptb", or "sst".')
    train_dataloader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
    val_dataloader = DataLoader(validation_set, batch_size=params['nb_val'], shuffle=False)


    model = RRNNforGRU(HIDDEN_SIZE, num_classes, batch_size=params['batch_size'],
                       scoring_hsize=params['scoring_hidden_size']).to(device)

    # Warm-start with pretrained GRU weights
    if params['pretrained_weights']:
        print('[INFO] Loading pre-trained GRU weights.')
        # Extract GRU pre-trained weights
        W_ir, W_iz, W_in = gru_model.weight_ih_l0.chunk(3)
        W_hr, W_hz, W_hn = gru_model.weight_hh_l0.chunk(3)
        b_ir, b_iz, b_in = gru_model.bias_ih_l0.chunk(3)
        b_hr, b_hz, b_hn = gru_model.bias_hh_l0.chunk(3)

        L1 = W_ir
        R1 = W_hr
        b1 = b_ir + b_hr
        L2 = W_iz
        R2 = W_hz
        b2 = b_iz + b_hz
        L3 = W_in
        R3 = W_hn
        b3 = b_in #+ r*b_hn

        model.cell.L_list[1] = nn.Parameter(L1)
        model.cell.L_list[2] = nn.Parameter(L2)
        model.cell.L_list[3] = nn.Parameter(L3)
        model.cell.R_list[1] = nn.Parameter(R1)
        model.cell.R_list[2] = nn.Parameter(R2)
        model.cell.R_list[3] = nn.Parameter(R3)
        model.cell.b_list[1] = nn.Parameter(b1)
        model.cell.b_list[2] = nn.Parameter(b2)
        model.cell.b_list[3] = nn.Parameter(b3)

    print('[INFO] Beginning training with %d training samples and %d '
          'validation samples.' % (len(train_set), len(validation_set)))
    trainer = RRNNTrainer(model, gru_model, train_dataloader, val_dataloader, params)
    loss_history, acc_history, structure_history = trainer.train(params['epochs'])
    pickle.dump(loss_history, open('loss_history.pkl', 'wb'))
    pickle.dump(acc_history, open('acc_history.pkl', 'wb'))
    pickle.dump(structure_history, open('structure_history.pkl', 'wb'))
    print()
    print('[INFO] Saved loss, accuracy, and structure history.')

    runtime = time.time() - start
    pickle.dump(runtime, open(RUNTIME_FILE, 'wb'))
    print('[INFO] Run complete. Runtime:', datetime.timedelta(seconds=runtime))

    torch.save(model.state_dict(), 'final_weights.pt')


def random_hyperparams():
    # Standard default parameters
    params = {
        'multiplier': 1,
        'nb_train': 10000,
        'nb_val': 1000,
        'validate_every': 1,
        'epochs': 1000,
        'epochs_per_checkpoint': 10,
        'pickle_every': 1,
        'optimizer': 'adam',
        'embeddings': 'gensim',
        'initial_train_mode': 'weights',
        'warm_start': False,
        'weights_file': '',
        'pretrained_weights': False,
        'device': 'cpu',
        'dataset': 'wiki',
        "max_time": 60 * 60   # 1 hour, should by around 100 epochs for batch size of 16
    }

    # Random hyperparameters
    params['learning_rate'] = 8 * 10 ** np.random.uniform(-4, -2)
    params['lambdas'] = (
        1,
        3 * 10 ** np.random.uniform(-3, -1),
        2 * 10 ** np.random.uniform(-6, -4),
        7 * 10 ** np.random.uniform(-4, -2),
    )
    params['loss2_margin'] = 10 ** np.random.uniform(-1, 0.5)
    params['scoring_hidden_size'] = int(10 ** np.random.uniform(2, 2.6))
    params['batch_size'] = int(10 ** np.random.uniform(1.5, 2.5))
    params['max_grad'] = 10 ** np.random.uniform(1.5, 2)
    params['alternate_every'] = int(10 ** np.random.uniform(0, 1))
    # TODO: Add temperature to this
    return params


def random_hyperparam_search(n_runs):
    # TODO: Make this generate output folders inside a "Runs" folder.
    print('[INFO] Beginning random hyperparameter search.')
    best_loss1 = np.Inf
    best_params = None
    for i in trange(n_runs):
        print('\n\n\n' + '-' * 80)
        dirname = 'run-' + str(time.time())
        os.mkdir(dirname)
        os.chdir(dirname)
        params = random_hyperparams()
        print('[INFO] Running the following hyperparameters:')
        pprint(params)
        run(params)

        # Get val loss
        val = []
        with open('val_loss.txt', 'r') as f:
            for iteration in f:
                val.append([float(term) for term in iteration.split()])
        val_loss = np.array(val)
        min_loss1 = np.min(val_loss[:, 0])

        if min_loss1 < best_loss1:
            best_loss1 = min_loss1
            best_params = params
        os.chdir('..')
        print('[INFO] Best hyperparameters so far:')
        pprint(best_params)
        print('[INFO] Lowest validation loss1 so far:', best_loss1)
    print('[INFO] Random hyperparameter search complete.')


if __name__ == '__main__':


    random_hyperparam_search(50)

    if platform.system() == 'Windows':
        dirname = 'test %s'%(time.asctime().replace(':', '_'))
        params = {
            "learning_rate": 1e-4,
            "multiplier": 1,
            "lambdas": [1, 0, 1e-8, 0.003],
            "nb_train": 1000,
            "nb_val": 200,
            "validate_every": 1,
            "epochs": 20,
            "loss2_margin": 1,
            "scoring_hidden_size": 64,
            "batch_size": 64,
            "epochs_per_checkpoint": 1,
            "optimizer": "adam",
            "embeddings": "gensim",
            "max_grad": 1,
            "initial_train_mode": "weights",
            "alternate_every": 1,
            "warm_start": False,
            "weights_file": "epoch_0.pt",
            "pretrained_weights": False,
            "device": "cpu",
        }
        if not params['warm_start']:
            os.mkdir(dirname)
        os.chdir(dirname)

        run(params)

    else: # elif platform.system() == '' # on server
        if len(sys.argv) != 3:
            raise Exception('Usage: python trainer.py <output_dir> <JSON parameter file>')
        dirname = sys.argv[1]
        param_file = sys.argv[2]
        with open(param_file, 'r') as f:
            params = json.load(f)

        if not params['warm_start']:
            os.mkdir(dirname)
        os.chdir(dirname)
        run(params)
