# Generates a JSON file containing randomly-sampled hyperparameters.

import sys
import json
import time
import numpy as np


# Standard default parameters
params = {
    'multiplier': 1,
    'nb_train': 10000,
    'nb_val': 5000,
    'validate_every': 1,
    'epochs': 1000,
    'epochs_per_checkpoint': 100,
    'pickle_every': 100,
    'optimizer': 'adam',
    'embeddings': 'gensim',
    'initial_train_mode': 'weights',
    'warm_start': False,
    'weights_file': '',
    'pretrained_weights': False,
    'device': 'cpu',
    'dataset': 'wiki'
}

# Random hyperparameters
params['learning_rate'] = 10 ** np.random.uniform(-5, -2)
params['lambdas'] = (
    1,
    10 ** np.random.uniform(-2, 1),
    10 ** np.random.uniform(-12, 0),
    10 ** np.random.uniform(-6, -3)
)
params['loss2_margin'] = 10 ** np.random.uniform(-1, 1)
params['scoring_hidden_size'] = int(10 ** np.random.uniform(0, 3))
params['batch_size'] = int(10 ** np.random.uniform(0, 3))
params['max_grad'] = 10 ** np.random.uniform(-1, 2)
params['alternate_every'] = int(10 ** np.random.uniform(0, 1))

# Write to JSON
if len(sys.argv) > 0:
    filename = sys.argv[1] + '.json'
else:
    filename = 'random_params_' + str(time.time()) + '.json'
json.dump(params, open(filename, 'w'), indent=4)
