import trainer
import time
import pickle
import torch
from GRU import RRNNforGRU
import dataloader

NB_DATA = 2
EPOCHS = 1
# RUNTIME = 5 * 24 * 60 * 60

def random_params():
    params = {}
    params['learning_rate'] = 10 ** np.random.uniform(-5, -2)
    params['multiplier'] = 10 ** np.random.uniform(-6, -2)
    lamb1 = 1
    lamb2 = 10 ** np.random.uniform(-3, 1)
    # lamb3 = 10 ** np.random.uniform(-3, 1)
    lamb3 = 0   # L2 regularization off for now
    lamb4 = 10 ** np.random.uniform(-1, 3)
    params['lambdas'] = (lamb1, lamb2, lamb3, lamb4)
    return params


if __name__ == '__main__':

    ### Random hyperparameter search ###

    max_gru_count = 0
    best_params = None

    start = time.time()

    # while (time.time() - start) < RUNTIME:
    print('='*80)
    print('\n[INFO] Beginning run.\n')
    # params = random_params()
    params = {
        'learning_rate': 1e-4,
        'multiplier': 1e-4,
        'lambdas': (1, 1e-2, 0, 50)
    }

    gru_model = torch.load('gru_parameters.pkl')
    model = RRNNforGRU(trainer.HIDDEN_SIZE, trainer.VOCAB_SIZE, params['multiplier'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    X_train, y_train = dataloader.load_normalized_data('train20.txt',
                                                       embeddings='gensim')
    for i in range(len(X_train)):
        X_train[i] = X_train[i].to(trainer.device)
        y_train[i] = torch.tensor(y_train[i], device=trainer.device)

    trainer = trainer.RRNNTrainer(model, gru_model, X_train[:NB_DATA], y_train[:NB_DATA],
                          optimizer, params['lambdas'])
    try:
        loss, gru_count = trainer.train(EPOCHS, verbose=True)
    except ValueError:
        print('ValueError')
        gru_count = -1

    pickle.dump(loss, open('loss.pkl', 'wb'))

    print('Hyperparameters:')
    print(params)
    print('\nAchieved the GRU structure on %d iterations.\n' % (gru_count,))
    if gru_count > max_gru_count:
        best_params = params
    print('Best hyperparameters so far:')
    print(best_params)
    print(flush=True)
