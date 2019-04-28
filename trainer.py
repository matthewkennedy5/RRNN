import os, sys, platform
import time
import datetime
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import time
import json

import tree_methods
from GRU import RRNNforGRU
import pickle
import pdb
from tqdm import tqdm
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


BATCH_HISTORY_DIR = 'batch_history/'

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

        if params['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=params['learning_rate'])
        elif params['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=params['learning_rate'])
        self.optimizer = optimizer

        self.params = params
        self.lamb1, self.lamb2, self.lamb3, self.lamb4 = params['lambdas']
        # self.loss = torch.nn.KLDivLoss()
        self.loss = torch.nn.CrossEntropyLoss()
        # TODO: Change this variable name--it's not a true iteration count. It
        # increments multiple times per batch.
        self.train_mode = params['initial_train_mode']
        
        # TODO: include the following in the heading part and params
        self.current_stage = 'searching' # another choice is 'fixing'
        if not os.path.isdir(BATCH_HISTORY_DIR):
            os.makedirs(BATCH_HISTORY_DIR)
        
    
    def load_optimal_history(self, i_batch):
        file_name = BATCH_HISTORY_DIR + '%d.txt'%i_batch
        
        if not os.path.isfile(file_name):
            raise ValueError('No such batch file')
        
        loss_history = []
        structure_history = []
        
        file = open(file_name, 'r')
        for line in file.readlines():
            i_epoch, i_batch, i_time_step, loss, structure = [eval(s) for s in line[:-1].split(';')]
            if i_epoch+1 > len(loss_history):
                loss_history.append([])
                structure_history.append([])
            loss_history[i_epoch].append(loss)
            structure_history[i_epoch].append(structure)
        file.close()
        
        loss_history = np.array(loss_history)
        idx = np.argmin(loss_history, axis=0)
        optimal_history = []
        for i_time_step in range(self.time_steps):
            optimal_history.append(structure_history[idx[i_time_step]][i_time_step])
            
        return optimal_history

    def switch_train_mode(self):
        """Switches the train mode and freezes parameters from the other mode.

        The training mode corresponds to which parameters we're trying to train.
        We are alternating between training the scoring NN and the L, R, b,
        output weights of the RRNN, since training them both at the same time
        interferes with each other.
        """
        if self.train_mode == 'weights':
            self.train_mode = 'scoring'
        else:
            self.train_mode = 'weights'
        print('[INFO] Switching to training the ' + self.train_mode + '.')
        self.freeze_params()

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
        for i_epoch in range(n_epochs):
            i_batch = -1
            print('\n[INFO] Epoch %d/%d' % (i_epoch+1, n_epochs))
            with tqdm(self.train_data) as t:
                for X_batch, y_batch in self.train_data:
                    i_batch += 1
                    
                    self.optimizer.zero_grad()
                    if self.current_stage == 'searching':
                        losses, acc, structures = self.train_step_stage_searching(X_batch, y_batch, i_epoch, i_batch)
                    else:   # elif self.current_stage == 'fixing':
                        losses, acc, structures = self.train_step_stage_fixing(X_batch, y_batch, i_epoch, i_batch)
                    loss_fn = sum(losses)
                    loss_fn.backward()
                    self.optimizer.step()

                    losses = printable(losses)
                    loss_history.append(losses)
                    acc_history.append(acc)
                    structure_history.append(structures)

                    t.set_postfix(loss=losses)
                    t.update()
            
            if i_epoch % self.params['validate_every'] == 0:
                if len(self.val_data) > 0:
                    self.validate(i_epoch)

            if i_epoch % self.params['epochs_per_checkpoint'] == 0:
                self.checkpoint_model(i_epoch+1)

            if i_epoch % self.params['pickle_every'] == 0:
                pickle.dump(loss_history, open('loss_history.pkl', 'wb'))
                pickle.dump(acc_history, open('acc_history.pkl', 'wb'))
                pickle.dump(structure_history, open('structure_history.pkl', 'wb'))
                print('[INFO] Saved loss, accuracy, and structure history.')

#            if i_epoch % self.params['alternate_every'] == 0:
#                self.switch_train_mode()
            
            # TODO: switch between training stages
                
        self.model.eval()
        return loss_history, acc_history, structure_history

    def train_step_stage_fixing(self, X, y, i_epoch, i_batch):
        optimal_structure = self.load_optimal_history(i_batch)
        batch_size, time_steps, HIDDEN_SIZE = X.shape
        lamb1, lamb2, lamb3, lamb4 = self.params['lambdas']
        
        # forward pass of the model
        pred_chars_batch = self.model(X, optimal_structure)
        
        # calculate loss terms
        loss1 = 0
        for i_time in range(time_steps):
                loss1 += self.loss(pred_chars_batch[:, i_time, :], torch.argmax(y[:, i_time, :], dim=1))
        
        loss3 = 0
        for param in self.model.parameters():
            loss3 += param.norm()**2
        
        # report loss and accuracy
        losses = [lamb1*loss1, lamb3*loss3]
        accuracy = (pred_chars_batch.argmax(dim=2)==y.argmax(dim=2)).sum().item()/float(time_steps*y.shape[0])
        
        return losses, accuracy, optimal_structure

    def train_step_stage_searching(self, X, y, i_epoch, i_batch):
        batch_size, time_steps, HIDDEN_SIZE = X.shape
        # forward pass
        pred_chars_batch, h_batch, pred_tree_list, structures_list, margins_batch = self.model(X)
        
        # forward pass of traditional GRU
        gru_h_list = self.gru_model(X)[0]
        gru_h_list = torch.cat([torch.zeros([batch_size, 1, HIDDEN_SIZE], device=X.device), gru_h_list], dim=1)
        target_tree_list = []
        for t in range(X.shape[1]):
            gru_x = X[:, t, :].reshape(batch_size, 1, HIDDEN_SIZE)
            gru_h = gru_h_list[:, t, :].reshape(batch_size, 1, HIDDEN_SIZE)
            target_tree = tree_methods.GRUtree_pytorch(gru_x, gru_h,
                                                       self.gru_model.weight_ih_l0,
                                                       self.gru_model.weight_hh_l0,
                                                       self.gru_model.bias_ih_l0,
                                                       self.gru_model.bias_hh_l0)[1]
            target_tree_list.append(target_tree)
            
        # get weight for each loss terms
        lamb1, lamb2, lamb3, lamb4 = params['lambdas']
        
        # calculate loss terms
        loss1_list = []
        for i_time in range(y.shape[1]):
            loss1_list.append(self.loss(pred_chars_batch[:, i_time, :], torch.argmax(y[:, i_time, :], dim=1)))
        loss1 = sum(loss1_list)

        loss2 = 0
        if lamb2 != 0:
            desired_margin = params['loss2_margin']
            loss2 = (desired_margin - margins_batch.clamp(max=desired_margin)).sum().div_(desired_margin)
            
        loss3 = 0
        if lamb3 != 0:
            for param in self.model.parameters():
                loss3 += param.norm()**2
        
        loss4 = 0
        if lamb4 != 0:
            loss4_list = []
            for i_time_step in range(time_steps):
                loss4_list.append(tree_methods.tree_distance_metric_list(
                                            pred_tree_list[i_time_step], 
                                            target_tree_list[i_time_step]))
            loss4 = sum(loss4_list)
                
        losses = [lamb1*loss1, lamb2*loss2, lamb3*loss3, lamb4*loss4]
        accuracy = (pred_chars_batch.argmax(dim=2)==y.argmax(dim=2)).sum().item()/float(time_steps*y.shape[0])

        # save batch history
        if isinstance(i_epoch, int):    # train
            file = open(BATCH_HISTORY_DIR+'%d.txt'%i_batch, 'a')
        elif i_epoch.startswith('val'): # val
            file = open(BATCH_HISTORY_DIR+'%s.txt'%i_epoch, 'a')
        else:
            raise ValueError
        for i_time_step in range(time_steps):
            lst = [i_epoch, i_batch, i_time_step, loss1_list[i_time_step].item(), structures_list[i_time_step]]
            file.write(';'.join([str(s) for s in lst])+'\n')
        file.close()
        
        return losses, accuracy, structures_list
    
    # TODO: fix it 
    def validate(self, i_epoch, verbose=True):
        """Runs inference over the validation set periodically during training.

        Prints the validation loss and accuracy to their respective files.
        """
        X_val, y_val = next(iter(self.val_data))
        val_size, time_steps, hidden_size = X_val.shape
        losses = np.zeros([val_size, N_LOSS_TERMS])
        accuracy = []

        for i_batch in range(val_size):
            X = X_val[i_batch, :, :].reshape(1, time_steps, hidden_size)
            y = y_val[i_batch, :, :].reshape(1, time_steps, -1)
            loss, acc, _ = self.train_step_stage_searching(X, y, 'val%d'%i_epoch, i_batch)
            losses[i_batch, :] = loss
            accuracy.append(acc)
            
        losses = losses.mean(axis=0).tolist()
        accuracy = np.mean(accuracy)

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
    # Assuming we are already in the directory where the output files should be
    pickle.dump(params, open(HYPERPARAM_FILE, 'wb'))
    print('[INFO] Saved hyperparameters.')

    start = time.time()
    device = torch.device(params['device'])
    gru_model = torch.load('../gru_parameters.pkl').to(device)

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

    model = RRNNforGRU(HIDDEN_SIZE, VOCAB_SIZE, batch_size=params['batch_size'],
                       scoring_hsize=params['scoring_hidden_size']).to(device)

    # Warm-start with pretrained GRU weights
    if params['pretrained_weights']:
        print('[INFO] Loading pre-trained GRU weights.')
        model.cell.L_list[1] = nn.Parameter(L1)
        model.cell.L_list[2] = nn.Parameter(L2)
        model.cell.L_list[3] = nn.Parameter(L3)
        model.cell.R_list[1] = nn.Parameter(R1)
        model.cell.R_list[2] = nn.Parameter(R2)
        model.cell.R_list[3] = nn.Parameter(R3)
        model.cell.b_list[1] = nn.Parameter(b1)
        model.cell.b_list[2] = nn.Parameter(b2)
        model.cell.b_list[3] = nn.Parameter(b3)

    if params['warm_start']:
        weights = params['weights_file']
        print('[INFO] Warm starting from ' + weights + '.')
        model.load_state_dict(torch.load(weights))

    print('[INFO] Loading training data into memory.')
    # TODO: Include other datasets
    train_set = standard_data.EnWik8Clean(subset='train', n_data=params['nb_train'], device=device)
    validation_set = standard_data.EnWik8Clean(subset='val', n_data=params['nb_val'], device=device)
    train_dataloader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=False, drop_last=True)
    val_dataloader = DataLoader(validation_set, batch_size=params['nb_val'], shuffle=False)
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


if __name__ == '__main__':


    if platform.system() == 'Windows':
        dirname = 'test %s'%(time.asctime().replace(':', '_'))
        params = {
                    "learning_rate": 1e-4,
                    "multiplier": 1,
                    "lambdas": [1, 0, 1e-8, 0.003],
                    "nb_train": 64,
                    "nb_val": 10,
                    "validate_every": 1,
                    "epochs": 2,
                    "loss2_margin": 1,
                    "scoring_hidden_size": 64,
                    "batch_size": 64,
                    "epochs_per_checkpoint": 1,
                    "pickle_every": 1,
                    "optimizer": "adam",
                    "embeddings": "gensim",
                    "max_grad": 1,
                    "initial_train_mode": "weights",
                    "alternate_every": 1,
                    "warm_start": False,
                    "weights_file": "epoch_0.pt",
                    "pretrained_weights": False,
                    "device": "cpu"
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

