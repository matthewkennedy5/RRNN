from pprint import pprint
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, Softmax
from tensorflow.keras.regularizers import l2
import torch
import pdb
from matplotlib import pyplot as plt
import numpy as np
import standard_data
from tqdm import trange

FILENAME = 'enwik8_clean.txt'
CHUNK_LENGTH = 20
# EPOCHS = 10
HIDDEN_SIZE = 100
N_CHARS = 27
# BATCH_SIZE = 16
# LEARNING_RATE = 1e-2
# L2_WEIGHT_DECAY = 0
DTYPE = np.float32
N_RANDOM_SEARCH = 100


print('[INFO] Loading data.')
data = standard_data.load_standard_data()
(X_train, y_train), (X_val, y_val), (X_test, y_test) = data

X_train = X_train.numpy().astype(DTYPE)
y_train = y_train.numpy()
X_val = X_val.numpy().astype(DTYPE)
y_val = y_val.numpy()
X_test = X_test.numpy().astype(DTYPE)
y_test = y_test.numpy()


def random_params():
    reg = 10 ** np.random.uniform(-8, -6)
    learning_rate = 10 ** np.random.uniform(-4, -1)
    epochs = 20
    batch_size = int(2 ** np.random.uniform(3, 5))
    params = {'reg': reg, 'learning rate': learning_rate, 'epochs': epochs,
              'batch size': batch_size}
    return params

def run(params):
    model = Sequential([
                GRU(HIDDEN_SIZE, return_sequences=True,
                    input_shape=(CHUNK_LENGTH, X_train.shape[2]),
                    kernel_regularizer=l2(params['reg']),
                    recurrent_regularizer=l2(params['reg'])),
                TimeDistributed(Dense(N_CHARS, kernel_regularizer=l2(params['reg']))),
                Softmax()
            ])

    adam = tf.keras.optimizers.Adam(lr=params['learning rate'])
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
    history = model.fit(X_train, y_train, batch_size=params['batch size'], epochs=params['epochs'],
                        validation_data=(X_val, y_val))
    return history


def random_hyperparam_search(n_runs):
    best_params = None
    lowest_val_loss = np.Inf
    for i in trange(N_RANDOM_SEARCH):
        params = random_params()
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

    params = {'batch size': 10,
              'epochs': 40,
              'learning rate': 0.0005193665291051191,
              'reg': 7.017320195906407e-07}

    history = run(params)
    # plot_results(history)
    # plot_bpc(history)
    # random_hyperparam_search(100)



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
#  'learning rate': 0.0017111124938516654,
#  'reg': 3.5989197176496724e-07}
