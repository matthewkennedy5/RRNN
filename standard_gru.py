import os
import string
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import load_embeddings
from tqdm import tqdm
import pdb
import pickle
from dataloader import word2vec_model

# Hyperparameters
BATCH_SIZE = 64
HIDDEN_SIZE = 100
CHUNK_LENGTH = 20  # How many characters to look at at a time
ITERATIONS = int(5e5)
LEARNING_RATE = 1e-4
DECAY_RATE = 0.9    # Multiply the learning rate by this every so often
DECAY_EVERY = 1000 # Decay the learning rate after this many iterations

# Filenames
FILENAME = 'enwik8_clean.txt'
SAVE_NAME = 'standard_gru_weights.pt'

EMBEDDINGS = word2vec_model
VOCABULARY = string.ascii_lowercase + ' '
CUDA = torch.cuda.is_available()

EMBED_SIZE = 100
PRINT_EVERY = 100
SAVE_EVERY = 1000
WARM_START = False   # Keep training the existing model instead of just using it
                    # to write out text

if CUDA:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def embed(text):
    """Perform character embedding on the given string.

    Input:
        text - String of characters which have defined embeddings

    Returns:
        Tensor of (len(text), EMBED_SIZE) containing the embeddings for each
        character.
    """
    result = torch.zeros(len(text), EMBED_SIZE, device=device)
    for i, ch in enumerate(text):
        if ch not in EMBEDDINGS:
            raise ValueError('Embeddings are not defined for the character "%s"' % (ch,))
        result[i, :] = torch.tensor(EMBEDDINGS[ch])
    return result


def batch_generator(filename, batch_size, chunk_length, iterations):
    """Generate a single batch of training data.

    Each batch consists of multiple randomly-chosen excerpts of the text in the
    given file. Since the RNN is learning to predict characters, y is shifted to
    the right of x by one index.

    Inputs:
        filename - File containing the training text
        batch_size - Number of training examples per batch
        chunk_length - Number of characters per training example
        iterations - How many batches to train for

    Yields:
        i - Iteration number
        x - Batch of training data
        y - Vocabulary indices of the correct characters. y[n] corresponds to
                x[n-1].
    """
    with open(filename) as f:
        text = f.read()
    for i in tqdm(range(iterations)):
        x = torch.empty(batch_size, chunk_length, EMBED_SIZE, device=device)
        y = torch.empty(batch_size, chunk_length, device=device)
        for b in range(batch_size):
            # Randomly select a chunk of text and embed it to a tensor.
            start = np.random.randint(0, len(text) - chunk_length - 1)
            chunk = text[start : start+chunk_length+1]
            x[b, :, :] = embed(chunk[:-1])
            for c, ch in enumerate(chunk[1:]):
                y[b, c] = VOCABULARY.index(ch)
        yield i, x, y.long()



class Trainer:
    """Class to handle training of the RNN model.

    Constructor inputs:
        model - The RNN model to train
        gen - Batch generator instance to generate training data
    """
    def __init__(self, model, gen):
        self.model = model
        self.gen = gen
        self.criterion = nn.CrossEntropyLoss()

    def train(self, learning_rate, batch_size, iters):
        """Train the model for the given number of iterations.

        Inputs:
            learning_rate
            batch_size
            iters - Number of iterations to train for

        Returns:
            loss_history - Numpy array containing the loss at each iteration.
        """
        loss_history = np.zeros(iters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, DECAY_RATE)
        try:
            for i, x, y in self.gen:
                y_pred = self.model(x)
                loss_fn = 0
                for j in range(CHUNK_LENGTH):
                    loss_fn += self.criterion(y_pred[:, j, :], y[:, j])

                optimizer.zero_grad()
                loss_fn.backward()
                optimizer.step()
                loss_history[i] = loss_fn.item()

                if i % SAVE_EVERY == 0:
                    torch.save(model.state_dict(), SAVE_NAME)
                    print('[INFO] Saved weights.')

                # if i % DECAY_EVERY == 0:
                #     scheduler.step()
                #     print('[INFO] Decayed the learning rate.')

                if i % 5 == 0:
                    print('\r', loss_fn.item())


        except KeyboardInterrupt:
            print('\n[INFO] Saving weights before quitting.\n')
            torch.save(model.state_dict(), SAVE_NAME)
            # Get rid of trailing zeros
            loss_history = loss_history[loss_history != 0]
            plt.figure()
            plt.plot(range(loss_history.shape[0]), loss_history)
            plt.savefig('loss.png')
            raise KeyboardInterrupt

        return loss_history


class StandardGRU(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(StandardGRU, self).__init__()
        self.output_size = output_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.i2o = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hiddens = self.gru(x)[0]
        batch_size, sequence_length, hidden_size = x.size()
        y_pred = torch.zeros(batch_size, sequence_length, self.output_size)
        for i in range(sequence_length):
            y_pred[:, i, :] = self.i2o(hiddens[:, i, :])
        return y_pred


if __name__ == '__main__':

    # model = SequenceGRU(EMBED_SIZE, HIDDEN_SIZE, len(VOCABULARY)).to(device)
    # model = nn.GRU(input_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True)
    model = StandardGRU(input_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, output_size=len(VOCABULARY)).to(device)
    gen = batch_generator(FILENAME, BATCH_SIZE, CHUNK_LENGTH, ITERATIONS)
    trainer = Trainer(model, gen)
    loss = trainer.train(LEARNING_RATE, BATCH_SIZE, ITERATIONS)
    torch.save(model.state_dict(), SAVE_NAME)
    plt.figure()
    plt.plot(range(ITERATIONS), loss)
    plt.savefig('loss.png')
