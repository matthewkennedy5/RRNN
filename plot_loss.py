# Simple script for plotting a loss history that has been saved
# as a pickle file.

import sys
import pickle
from matplotlib import pyplot as plt
import numpy as np


def plot(fname, save_name):
    loss = pickle.load(open(fname, 'rb'))
    index = np.sum(loss != 0)
    plt.figure()
    plt.plot(range(index), loss[:index])
    plt.savefig(save_name)


if __name__ == '__main__':
    fname = 'loss.pkl'
    save_name = sys.argv[1]
    plot(fname, save_name)