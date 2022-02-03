import numpy as np

def scaledIdentityCovGenerator(mean, cov, n_random_vectors):
    vecs = np.random.normal(0, cov[0, 0], (mean.shape[0], n_random_vectors)) + mean[:, np.newaxis]
    return vecs.T
