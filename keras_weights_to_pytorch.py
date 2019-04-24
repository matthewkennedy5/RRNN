import tensorflow.keras
import torch
import pdb
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('keras_gru_weights.h5')
gru = model.layers[0]
W, U, b = gru.get_weights()

W_z, W_r, W_h = np.split(W, 3, axis=1)
U_z, U_r, U_h = np.split(U, 3, axis=1)
b_z, b_r, b_h = np.split(b, 3, axis=0)
