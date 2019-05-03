import os
import time
import torch
import numpy as np
from tqdm import tqdm

import tree_methods


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
        s1 = params['stage_searching_epochs']
        s2 = params['stage_fixing_epochs']
        self.switching_time = sorted([(s1+s2)*i for i in range(1, 20)]+[(s1+s2)*i+s1 for i in range(20)])
#        self.switching_time = [2]

        self.time_steps = 20
        self.num_batches = params['nb_train']//params['batch_size']
        
    def load_optimal_history(self, current_epoch):
        decay_constant = self.params['structure_decaying_constant']
        optimal_structures = [[0 for i in range(self.time_steps)] for j in range(self.num_batches)]
        optimal_epochs = [[0 for i in range(self.time_steps)] for j in range(self.num_sbatches)]

        for i_batch in range(self.num_batches):
            for i_time_step in range(self.time_steps):
                history_file_name = self.params['STRUCTURE_HISTORY_DIR'] + 'structure_batch%d_timestep%d.txt'%(i_batch, i_time_step)
                if not os.path.isfile(history_file_name):
                    raise ValueError('No such batch file')
            
                opt_loss = 1e99
                with open(history_file_name) as f:
                    for line in f.readlines():
                        i_epoch, loss, structure = [eval(s) for s in line[:-1].split(';')]
                        if decay_constant != 1:
                            diff_stage = (current_epoch - i_epoch)//self.params['stage_fixing_epochs']
                            loss = loss * (decay_constant**diff_stage)
                        if loss < opt_loss:
                            opt_loss = loss
                            opt_epoch = i_epoch
                            opt_structure = structure
                optimal_structures[i_batch][i_time_step] = opt_structure
                optimal_epochs[i_batch][i_time_step] = opt_epoch
                
        optimal_file_name = self.params['STRUCTURE_OPTIMAL_DIR'] + 'search_at_epoch%d.txt'%current_epoch
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
        torch.save(self.model.state_dict(), self.params['CHECKPOINT_DIR']+save_name)
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
                        record_history(self.params['TRAIN_LOSS_FILE'], i_epoch, i_batch, printable(losses))
                        record_history(self.params['TRAIN_ACC_FILE'], i_epoch, i_batch, acc)
                    
                    t.set_postfix(loss=printable(losses))
                    t.update()

            if self.current_stage == 'searching':
                self.checkpoint_model(i_epoch+1)
                
            if self.current_stage == 'fixing' and (i_epoch+1) % 10 == 0:
                self.checkpoint_model(i_epoch+1)

            if (i_epoch+1) in self.switching_time:
                self.switch_train_stages()
                self.optimal_structures, self.optimal_epochs = self.load_optimal_history(i_epoch)

            if (self.params['write_every_epoch'] is True) and \
               (self.params['write_every_epoch'] is not True):
                record_history(self.params['TRAIN_LOSS_FILE'], i_epoch, i_batch, printable(losses))
                record_history(self.params['TRAIN_ACC_FILE'], i_epoch, i_batch, acc)
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
            loss1 += self.loss(pred_chars_batch[:, i_time, :], y[:, i_time])
        
        loss3 = 0
        for param in self.model.parameters():
            loss3 += param.norm()**2
        
        # report loss and accuracy
        losses = [lamb1*loss1, 0, lamb3*loss3, 0]
        accuracy = (pred_chars_batch.argmax(dim=2)==y).sum().item()/float(time_steps*X.shape[0])
        
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
        lamb1, lamb2, lamb3, lamb4 = self.params['lambdas']
        
        # calculate loss terms
        loss1_list = []
        for i_time in range(y.shape[1]):
            loss1_list.append(self.loss(pred_chars_batch[:, i_time, :], y[:, i_time]))
        loss1 = sum(loss1_list)

        loss2 = 0
        if lamb2 != 0:
            desired_margin = self.params['loss2_margin']
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
            with open(self.params['STRUCTURE_HISTORY_DIR']+'structure_batch%d_timestep%d.txt'%(i_batch, i_time_step), 'a') as f:
                lst = [i_epoch, loss1_list[i_time_step].item(), structures_list[i_time_step]]
                f.write(';'.join([str(s) for s in lst])+'\n')
        
        return losses, accuracy, structures_list
    
    def validate(self, i_epoch, verbose=True):
        """Runs inference over the validation set periodically during training.

        Prints the validation loss and accuracy to their respective files.
        """
        X_val, y_val = next(iter(self.val_data))
        val_size, time_steps, hidden_size = X_val.shape
        losses = np.zeros([val_size, self.params['N_LOSS_TERMS']])
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
        with open(self.params['VAL_LOSS_FILE'], 'a') as f:
            f.write('%f %f %f %f\n' % tuple(losses))
        f.close()
        with open(self.params['VAL_ACC_FILE'], 'a') as f:
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
    if type(values) is list:
        line = [i_epoch, i_batch] + values
    elif type(values) in [float, int]:     
        line = [i_epoch, i_batch, values]
    else:
        raise ValueError('Unsupported value format!')
        
    with open(filename, 'a') as f:
        f.write(';'.join([str(s) for s in line])+'\n')
    
    return None

