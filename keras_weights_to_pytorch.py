import tensorflow.keras
import torch
import pdb
import numpy as np
from tensorflow.keras.models import load_model
from standard_data import EnWik8Clean
from torch.utils.data import DataLoader
from torch import nn

model = load_model('weights.h5')
gru = model.layers[0]
W, U, b = gru.get_weights()

W_z, W_r, W_h = np.split(W, 3, axis=1)
U_z, U_r, U_h = np.split(U, 3, axis=1)
b_z, b_r, b_h = np.split(b, 3, axis=0)

pytorch_gru = torch.load('gru_parameters.pkl')

W = np.vstack((W_r, W_z, W_h))
U = np.vstack((U_r, U_z, U_h))
# PyTorch has redundant bias vectors, so we have to divide by 2 since it will
# be added twice in the same equation.
b = np.hstack((b_r, b_z, b_h)) / 2
W = nn.Parameter(torch.tensor(W))
U = nn.Parameter(torch.tensor(U))
b_i = nn.Parameter(torch.tensor(b))
b_h = nn.Parameter(torch.tensor(b))

pytorch_gru.weight_ih_l0 = W
pytorch_gru.weight_hh_l0 = U
pytorch_gru.bias_ih_l0 = b_i
pytorch_gru.bias_hh_l0 = b_h

torch.save(pytorch_gru, 'gru_parameters_new.pt')


if __name__ == '__main__':

    dataset = EnWik8Clean('val', 10000, device='cpu')
    dataloader = DataLoader(dataset, batch_size=10000, shuffle=False)
    X_val, y_val = next(iter(dataloader))

    # Accuracy should be chance because there's no dense layer at the end
    # mapping hidden states to character predictions.
    out, _ = pytorch_gru(X_val)
    accuracy = (torch.argmax(out, dim=2) == y_val).sum().float() / (20 * 10000)
    print('accuracy:', accuracy.item())

