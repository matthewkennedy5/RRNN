import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, Softmax
import torch
import dataloader
import pdb
from matplotlib import pyplot as plt

FILENAME = 'enwik8_clean.txt'
CHUNK_LENGTH = 20
N_TRAIN = 55000
EPOCHS = 40
HIDDEN_SIZE = 100
N_CHARS = 27
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

print('[INFO] Loading data.')
X_train, y_train, _, _ = dataloader.load_normalized_data(filename=FILENAME,
                                                         chunk_length=CHUNK_LENGTH,
                                                         n_train=N_TRAIN)
X_train = X_train.squeeze()
y_train = y_train.long()
X_train = X_train.numpy()
y_train = y_train.numpy()

model = Sequential([
            GRU(HIDDEN_SIZE, return_sequences=True, input_shape=(CHUNK_LENGTH, X_train.shape[2])),
            TimeDistributed(Dense(N_CHARS)),
            Softmax()
        ])

adam = tf.keras.optimizers.Adam(lr=LEARNING_RATE)
model.compile(optimizer=adam, loss='categorical_crossentropy')
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure()
plt.plot(range(len(loss)), loss, label='Training loss')
plt.plot(range(len(val_loss)), val_loss, label='Validation loss')
plt.ylabel('Loss per character')
plt.xlabel('Epoch')
plt.title('Loss vs. epochs for 50k training examples')
plt.legend()
plt.savefig('50kloss.png')

