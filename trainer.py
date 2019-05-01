import os, sys, platform
import time
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import json

import tree_methods
from GRU import RRNNforGRU
import pickle
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

STRUCTURE_HISTORY_DIR = 'structure_history/'
STRUCTURE_OPTIMAL_DIR = 'structure_optimal/'
CHECKPOINT_DIR = 'checkpoints/'    

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
        self.loss = torch.nn.CrossEntropyLoss()

        self.current_stage = 'searching' # another choice is 'fixing'
        self.switching_time = [20, 1020, 1040, 2040, 2060, 3060, 3080, 4080]
        
        self.time_steps = 20
        self.num_batches = params['nb_train']//params['batch_size']
        
    def load_optimal_history(self, i_epoch):
        optimal_structures = [[0 for i in range(self.time_steps)] for j in range(self.num_batches)]
        optimal_epochs = [[0 for i in range(self.time_steps)] for j in range(self.num_batches)]

        for i_batch in range(self.num_batches):
            for i_time_step in range(self.time_steps):
                history_file_name = STRUCTURE_HISTORY_DIR + 'structure_%d_%d.txt'%(i_batch, i_time_step)
                if not os.path.isfile(history_file_name):
                    raise ValueError('No such batch file')
            
                opt_loss = 1e99
                with open(history_file_name) as f:
                    for line in f.readlines():
                        i_epoch, loss, structure = [eval(s) for s in line[:-1].split(';')]
                        if loss < opt_loss:
                            opt_loss = loss
                            opt_epoch = i_epoch
                            opt_structure = structure
                optimal_structures[i_batch][i_time_step] = opt_structure
                optimal_epochs[i_batch][i_time_step] = opt_epoch
                
        optimal_file_name = STRUCTURE_OPTIMAL_DIR + 'search_epoch_%d.txt'%i_epoch
        with open(optimal_file_name, 'a') as f:
            for i_batch in range(self.num_batches):
                for i_time_step in range(self.time_steps):
                    line = [optimal_epochs[i_batch][i_time_step], i_batch, i_time_step, optimal_structures[i_batch][i_time_step]]
                    f.write(';'.join([str(s) for s in line])+'\n')
                         
        return optimal_structures, optimal_epochs

    def switch_train_stages(self):
        """
            Switches the train stages 
        """
        if self.current_stage == 'searching':
            self.current_stage = 'fixing'
        elif self.current_stage == 'fixing':
            self.current_stage = 'searching'
        else:   
            raise ValueError('wrong stage mode')
            
        print('[INFO] Switching to training Stage: ' + self.current_stage+ '.')
        return None

    def checkpoint_model(self, i_epoch):
        save_name = 'checkpoint_' + str(i_epoch) + '_' + time.asctime().replace(':', ' ') + '.pt'
        torch.save(self.model.state_dict(), CHECKPOINT_DIR+save_name)
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

                    if self.params['write_every_batch'] is True:
                        record_history(TRAIN_LOSS_FILE, i_epoch, i_batch, printable(losses))
                        record_history(TRAIN_ACC_FILE, i_epoch, i_batch, acc)
                    
                    t.set_postfix(loss=printable(losses))
                    t.update()

            if i_epoch % self.params['epochs_per_checkpoint'] == 0:
                self.checkpoint_model(i_epoch+1)

            if (i_epoch+1) in self.switching_time:
                self.switch_train_stages()
                self.optimal_structures, self.optimal_epochs = self.load_optimal_history(i_epoch)

            if (self.params['write_every_epoch'] is True) and \
               (self.params['write_every_epoch'] is not True):
                record_history(TRAIN_LOSS_FILE, i_epoch, i_batch, printable(losses))
                record_history(TRAIN_ACC_FILE, i_epoch, i_batch, acc)
                print('[INFO] Saved loss, accuracy, and structure history.')                
            
        self.model.eval()
        return None

    def train_step_stage_fixing(self, X, y, i_epoch, i_batch):
        optimal_structure = self.optimal_structures[i_batch]
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
            loss1_list.append(self.loss(pred_chars_batch[:, i_time, :], y[:, i_time]    )
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
        accuracy = (pred_chars_batch.argmax(dim=2)==y).sum().item()/float(time_steps*X.shape[0])

        # save batch structure history
        for i_time_step in range(time_steps):
            with open(STRUCTURE_HISTORY_DIR+'structure_%d_%d.txt'%(i_batch, i_time_step), 'a') as f:
                lst = [i_epoch, loss1_list[i_time_step].item(), structures_list[i_time_step]]
                f.write(';'.join([str(s) for s in lst])+'\n')
        
        return losses, accuracy, structures_list
    
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

def record_history(filename, i_epoch, i_batch, values):
    if not filename in [TRAIN_ACC_FILE, TRAIN_LOSS_FILE]:
        raise ValueError('Unknown filename!')
        
    if type(values) is list:
        line = [i_epoch, i_batch] + values
    elif type(values) in [float, int]:     
        line = [i_epoch, i_batch, values]
    else:
        raise ValueError('Unsupported value format!')
        
    with open(filename, 'a') as f:
        f.write(';'.join([str(s) for s in line])+'\n')
    
    return None

# Performs a training run using the given hyperparameters. Saves out data and model checkpoints
# into the current directory.
def run(params):
    # Assuming we are already in the directory where the output files should be
    pickle.dump(params, open(HYPERPARAM_FILE, 'wb'))
    print('[INFO] Saved hyperparameters.')

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
    train_set = standard_data.EnWik8Clean(subset='train', n_data=params['nb_train'], device=device)
    validation_set = standard_data.EnWik8Clean(subset='val', n_data=params['nb_val'], device=device)
    train_dataloader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=False, drop_last=True)
    val_dataloader = DataLoader(validation_set, batch_size=params['nb_val'], shuffle=False)
    print('[INFO] Beginning training with %d training samples and %d '
          'validation samples.' % (len(train_set), len(validation_set)))

    trainer = RRNNTrainer(model, gru_model, train_dataloader, val_dataloader, params)
    trainer.train(params['epochs'])
    print()
    print('[INFO] Run complete')

    torch.save(model.state_dict(), 'final_weights.pt')
    return trainer

if __name__ == '__main__':

    if platform.system() in ['Windows', 'Darwin']:
        dirname = 'test %s'%(time.asctime().replace(':', '_'))
        params = {
                    "learning_rate": 1e-4,
                    "multiplier": 1,
                    "lambdas": [1, 0, 1e-8, 0.003],
                    "nb_train": 128,
                    "nb_val": 10,
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
                    'write_every_epoch': True,
                    'write_every_batch': True
                }
        if not params['warm_start']:
            os.mkdir(dirname)
        os.chdir(dirname)
        
        for path in [STRUCTURE_HISTORY_DIR, STRUCTURE_OPTIMAL_DIR, CHECKPOINT_DIR]:
            if not os.path.isdir(path):
                os.makedirs(path)
            
        trainer = run(params)
        model = trainer.model

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
        
        for path in [STRUCTURE_HISTORY_DIR, STRUCTURE_OPTIMAL_DIR, CHECKPOINT_DIR]:
            if not os.path.isdir(path):
                os.makedirs(path)
        
        trainer = run(params)
        model = trainer.model

