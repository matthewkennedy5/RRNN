import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, Softmax
import torch
import pdb
from matplotlib import pyplot as plt
import numpy as np
import standard_data

FILENAME = 'enwik8_clean.txt'
CHUNK_LENGTH = 20
EPOCHS = 100
HIDDEN_SIZE = 100
N_CHARS = 27
BATCH_SIZE = 64
LEARNING_RATE = 1e-2

print('[INFO] Loading data.')
data = standard_data.load_standard_data()
(X_train, y_train), (X_val, y_val), (X_test, y_test) = data

X_train = X_train.numpy()
y_train = y_train.numpy()
X_val = X_val.numpy()
y_val = y_val.numpy()
X_test = X_test.numpy()
y_test = y_test.numpy()

model = Sequential([
            GRU(HIDDEN_SIZE, return_sequences=True, input_shape=(CHUNK_LENGTH, X_train.shape[2])),
            TimeDistributed(Dense(N_CHARS)),
            Softmax()
        ])

adam = tf.keras.optimizers.Adam(lr=LEARNING_RATE)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_data=(X_val, y_val))

# Calculate and plot BPC
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
plt.show()

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

