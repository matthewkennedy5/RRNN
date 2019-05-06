import os
from pprint import pprint
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, Softmax
from tensorflow.keras.regularizers import l2
import pdb
import torch
from matplotlib import pyplot as plt
import numpy as np
import standard_data
from tqdm import trange
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '1'    # To select which GPU to use.
FILENAME = 'enwik8_clean.txt'
CHUNK_LENGTH = 20
HIDDEN_SIZE = 100
DTYPE = np.float32
N_RANDOM_SEARCH = 100
DEVICE = torch.device('cpu')
N_TRAIN = 10000
N_VAL = 10000


def load_data(dataset):
    print('[INFO] Loading the', dataset, 'dataset.')
    if dataset == 'ptb':
        train_data = standard_data.PennTreebank('train', n_data=N_TRAIN, device=DEVICE)
        train_dataloader = DataLoader(train_data, batch_size=len(train_data))
        val_data = standard_data.PennTreebank('val', n_data=N_VAL, device=DEVICE)
        val_dataloader = DataLoader(val_data, batch_size=len(val_data))
        X_train = next(iter(train_dataloader))[0].numpy()
        y_train = next(iter(train_dataloader))[1].numpy()
        X_val = next(iter(val_dataloader))[0].numpy()
        y_val = next(iter(val_dataloader))[1].numpy()
    elif dataset == 'wiki':
        data = standard_data.load_standard_data()
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data
        X_train = X_train.numpy().astype(DTYPE)
        y_train = y_train.numpy()
        X_val = X_val.numpy().astype(DTYPE)
        y_val = y_val.numpy()
        X_test = X_test.numpy().astype(DTYPE)
        y_test = y_test.numpy()
    return (X_train, y_train), (X_val, y_val)


def random_params(dataset):
    reg = 10 ** np.random.uniform(-16, 0)
    learning_rate = 10 ** np.random.uniform(-4, -1)
    epochs = 5
    batch_size = int(2 ** np.random.uniform(3, 8))
    hidden_size = 100
    params = {'reg': reg, 'learning_rate': learning_rate, 'epochs': epochs,
              'batch_size': batch_size, 'hidden_size': hidden_size, 'dataset': dataset}
    return params


def run(params):
    (X_train, y_train), (X_val, y_val) = load_data(params['dataset'])
    if params['dataset'] == 'wiki':
        n_out = 27
    elif params['dataset'] == 'ptb':
        n_out = 10001
    model = Sequential([
                GRU(params['hidden_size'], return_sequences=True,
                    input_shape=(CHUNK_LENGTH, X_train.shape[2]),
                    kernel_regularizer=l2(params['reg']),
                    recurrent_regularizer=l2(params['reg'])),
                TimeDistributed(Dense(n_out, kernel_regularizer=l2(params['reg']))),
                Softmax()
            ])

    adam = tf.keras.optimizers.Adam(lr=params['learning_rate'])
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['acc'])
    y_train = np.expand_dims(y_train, 2)
    y_val = np.expand_dims(y_val, 2)
    history = model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                        validation_data=(X_val, y_val))
    # model.save('weights.h5')
    return history


def random_hyperparam_search(n_runs, dataset):
    best_params = None
    lowest_val_loss = np.Inf
    for i in trange(N_RANDOM_SEARCH):
        params = random_params(dataset)
        print('-'*70)
        print('[INFO] Trying the following hyperparameters:')
        pprint(params)
        history = run(params)
        print('[INFO] Run complete.')
        val_loss = np.min(history.history['val_loss'])
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_params = params
        print('[INFO] Best validation loss achieved for this run: %f' % (val_loss,))
        print('[INFO] Best hyperparameters found so far: ')
        pprint(best_params)
        print('[INFO] Best validation loss achieved so far: ' + str(lowest_val_loss))

    print('[INFO] Random search complete.')
    print('[INFO] Best hyperparameters found: ')
    pprint(best_params)


def plot_results(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.figure(figsize=(20, 15))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_loss)), train_loss, label='Train loss')
    plt.plot(range(len(val_loss)), val_loss, label='Validation loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(len(train_acc)), train_acc, label='Train accuracy')
    plt.plot(range(len(val_acc)), val_acc, label='Validation accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()


def plot_bpc(history):
    # Calculate and plot "BPC"
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    train_bpc = -np.log2(train_acc)
    val_bpc = -np.log2(val_acc)

    plt.figure()
    plt.plot(range(train_bpc.shape[0]), train_bpc, label='Training BPC')
    plt.plot(range(val_bpc.shape[0]), val_bpc, label='Validation BPC')
    plt.ylabel('Bits per character')
    plt.xlabel('Epoch')
    plt.title('BPC vs Epochs')
    plt.legend()
    plt.savefig('bpc.png')
    plt.show()


if __name__=='__main__':

    # Best wiki params
    # params = {
    #     'batch_size': 10,
    #     'epochs': 34,
    #     'learning_rate': 0.0005193665291051191,
    #     'reg': 7.017320195906407e-07,
    #     'hidden_size': 100,
    #  }

    params = {
        'batch_size': 16,
        'epochs': 5,
        'learning_rate': 1e-2,
        'reg': 1e-4,
        'hidden_size': 100,
        'dataset': 'ptb'
    }
    run(params)

    # random_hyperparam_search(100, dataset='ptb')



# Plot loss
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# plt.figure()
# plt.plot(range(len(loss)), loss, label='Training loss')
# plt.plot(range(len(val_loss)), val_loss, label='Validation loss')
# plt.ylabel('Loss per character')
# plt.xlabel('Epoch')
# plt.title('Loss vs. epochs')
# plt.legend()
# plt.show()
# plt.savefig('loss.png')

# [INFO] Best hyperparameters found:
# {'batch size': 18,
#  'epochs': 10,
#  'learning_rate': 0.0017111124938516654,
#  'reg': 3.5989197176496724e-07}
