import os
import torch
import pickle
import numpy as np
import pdb
from tqdm import tqdm
from gensim.models import Word2Vec
from torch.utils import data
from torchtext import data, datasets
from dataloader import element_dict

dirname = os.path.dirname(os.path.realpath(__file__))
CORPUS_FILENAME = os.path.join(dirname, 'enwik8_clean.txt')
SAVE_FILE = os.path.join(dirname, '10k_data.pkl')
CHUNK_LENGTH = 20
EMBEDDINGS = 'gensim'
N_TRAIN = 10000
N_VAL = 1000
N_TEST = 2000
N_CHARS = len(element_dict)

# Load the embed dictionary that maps characters to vectors
word2vec_model = Word2Vec.load('train20_embedding')
if EMBEDDINGS == 'magic':
    EMBED = load_embeddings.load(MAGIC_EMBEDDINGS_FILE)
else:
    EMBED = word2vec_model
EMBED_SIZE = EMBED['a'].shape[0]

def index_overlaps(index, other_indices):
    """Returns True if the given index overlaps with other_indices.

    An index overlaps with another index if the chunks they produce overlap.
    Each index corresponds to the first character in a chunk.

    Inputs:
        index - A number in the range of len(corpus)
        other_indices - List of the other indices we're checking
    """
    for i in other_indices:
        if abs(index - i) < CHUNK_LENGTH:
            return True     # i and index overlaps
    return False

def pick_indices(corpus_length):
    """Selects indices for the beginnings of chunks for each datum.

    The selected indices are guaranteed to not produce chunks that overlap with
    eachother. This guarantees that all samples have distinct data and that the
    train, test, and val sets are disjoint.

    Returns:
        train_indicies - Starting indices for train set chunks
        val_indices - Starting indices for validation set chunks
        test_indices - Starting indices for test set chunks.
    """
    n_data = N_TRAIN + N_VAL + N_TEST
    if n_data * CHUNK_LENGTH > corpus_length * 2:
        raise ValueError('Corpus is too small for the amount of data requested.')

    indices = []
    for i in tqdm(range(N_TRAIN + N_VAL + N_TEST)):
        while True:
            index = np.random.randint(corpus_length)
            if not index_overlaps(index, indices):
                indices.append(index)
                break

    train_indices = indices[:N_TRAIN]
    val_indices = indices[N_TRAIN:N_TRAIN+N_VAL]
    test_indices = indices[-N_TEST:]
    return train_indices, val_indices, test_indices


def load_chunk(corpus, start):
    """Translates text into usable x and y variables.

    Inputs:
        corpus - String containing the dataset
        start - Start index of the chunk in the dataset string

    Returns:
        x - Tensor containing the embedded representation for the chunk.
        y - Tensor of one-hot labels for the truth values at each time step.
    """
    word = []   # List of embedded character vectors for the chunk
    input_word = corpus[start:start+CHUNK_LENGTH]
    for j in range(len(input_word)):
        if input_word[j] == ' ':
            # TODO: What's the point of replacing spaces with periods?
            word.append(EMBED['.'])
        else:
            word.append(EMBED[input_word[j]])

    # Construct the one-hot truth labels corresponding to the next character
    # in the sequence at each time step.
    sentence_length = len(input_word)
    n_chars = len(element_dict)
    y = torch.zeros(sentence_length, n_chars)
    # The input_word but shifted one character in the future
    for j, ch in enumerate(corpus[start+1:start+CHUNK_LENGTH+1]):
        y[j, element_dict[ch]] = 1

    x = torch.tensor(word).reshape(1, -1, 100)
    return x, y


def partition_data():
    """Selects training, validation, and test samples from the corpus to save.

    This function writes a pickle files containing the train, val, and test
    sets stored as as a tuple of tuples of (X, y).
    """
    with open(CORPUS_FILENAME, 'r') as f:
        corpus = f.read()
        train_indices, val_indices, test_indices = pick_indices(len(corpus))

        X_train = torch.empty(N_TRAIN, CHUNK_LENGTH, EMBED_SIZE)
        y_train = torch.empty(N_TRAIN, CHUNK_LENGTH, N_CHARS)
        X_val = torch.empty(N_VAL, CHUNK_LENGTH, EMBED_SIZE)
        y_val = torch.empty(N_VAL, CHUNK_LENGTH, N_CHARS)
        X_test = torch.empty(N_TEST, CHUNK_LENGTH, EMBED_SIZE)
        y_test = torch.empty(N_TEST, CHUNK_LENGTH, N_CHARS)

        # Embed the chunks and store them their corresponding tensors
        for i, index in enumerate(train_indices):
            x, y = load_chunk(corpus, index)
            X_train[i, :, :] = x
            y_train[i, :, :] = y
        for i, index in enumerate(val_indices):
            x, y = load_chunk(corpus, index)
            X_val[i, :, :] = x
            y_val[i, :, :] = y
        for i, index in enumerate(test_indices):
            x, y = load_chunk(corpus, index)
            X_test[i, :, :] = x
            y_test[i, :, :] = y

        # Normalize the data based on the statistics of the training set
        mean = torch.mean(X_train, dim=0)
        std = torch.std(X_train, dim=0)
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std

        y_train = y_train.long()
        y_val = y_val.long()
        y_test = y_test.long()

        data = ((X_train, y_train),
               (X_val, y_val),
               (X_test, y_test))

        with open(SAVE_FILE, 'wb') as save_file:
            pickle.dump(data, save_file)


def load_standard_data():
    """Loads and returns the training, validation, and test data.

    The point of this method is that if we're comparing RRNN to GRU, we want to
    use the same data. So we specifically store the data that we use so we can
    use it to train and evaluate both models instead of randomly selecting samples.

    Returns:
        data - Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test)).
    """
    if not os.path.isfile(SAVE_FILE):
        print('[INFO] Partitioning fresh training, validation, and test data '
              'from the corpus.')
        partition_data()

    data = pickle.load(open(SAVE_FILE, 'rb'))
    return data


class EnWik8Clean(data.Dataset):
    """Dataset containing the enwik8_clean.txt 27-character Wikipedia data.

    Inputs:
        subset - either 'train', 'val', or 'test'.
        n_data - How many data points to sample (with a maximum determined by
            the size of the data in the pickle file.)
        device
    """

    def __init__(self, subset, n_data, device):
        train, val, test = load_standard_data()
        if subset == 'train':
            self.X, self.y = train
        elif subset == 'val':
            self.X, self.y = val
        elif subset == 'test':
            self.X, self.y = test
        else:
            raise ValueError('Subset input must be "train", "val", or "test"')

        self.X = self.X[:n_data].to(device)
        self.y = self.y[:n_data].to(device)
        self.y = torch.argmax(self.y, dim=2)  # Convert to index labels

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return (self.X[index], self.y[index])

class PennTreebank(data.Dataset):
    """Dataset containing the Penn Treebank word-level language modelling data.

    Inputs:
        subset - either 'train', 'val', or 'test'.
        n_data - how many data points to sample.
        device - torch.device on which to store the data
    """
    def __init__(self, subset, n_data, device):
        train, valid, test = datasets.PennTreebank.iters(batch_size=1,
                                                         bptt_len=CHUNK_LENGTH,
                                                         device=device,
                                                         vectors='glove.6B.100d')
        if subset == 'train':
            self.data = train
        elif subset == 'val':
            self.data = val
        elif subset == 'test':
            self.data = test
        self.embeddings = self.data.dataset.fields['text'].vocab.vectors

        self.X = []
        self.y = []
        for sample in self.data:
            # sample.text contains the index of the embedded vector. We store the
            # vectors themselves in self.X, and word indices in self.y.
            if len(sample.text) == CHUNK_LENGTH:    # To avoid the leftover chunk of 8 at the end
                self.X.append(self.embeddings[sample.text])
                self.y.append(sample.target)
        self.X = torch.stack(self.X).squeeze()
        self.y = torch.stack(self.y).squeeze()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return (self.X[index], self.y[index])

