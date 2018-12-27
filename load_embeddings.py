import numpy as np

DIMENSIONS = 300    # Size of embedded vectors

def load(file, dtype='float32'):
    # file should contain a trailing newline
    embeddings = {}
    with open(file) as f:
        for line in f:
            char = line[0]
            embeddings[char] = np.zeros(DIMENSIONS)
            for i, word in enumerate(line[1:].split()):
                embeddings[char][i] = float(word)
            embeddings[char] = embeddings[char].astype(dtype)
    return embeddings
