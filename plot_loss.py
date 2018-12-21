# Simple script for plotting a loss history that has been saved
# as a pickle file.

import sys
import pickle
from matplotlib import pyplot as plt
import numpy as np

fname = sys.argv[1]
loss = pickle.load(open(fname, 'rb'))
index = np.sum(loss != 0)
plt.figure()
plt.plot(range(index), loss[:index])
plt.show()
