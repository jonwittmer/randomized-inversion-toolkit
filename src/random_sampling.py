import numpy as np
from utils.math import isScalar

'''
    This helper function makes generating samples from an identity covariance more efficient while 
    creating a uniform interface for factorizing a covariance matrix. When cov is a scalar, 
    this function returns
    U = Identity
    D = cov * [1, 1, ..., 1]
    V = Identity
'''
def covarianceSvd(vector_shape, cov):
    if isScalar(cov):
        U, D, Vt = np.eye(vector_shape), cov * np.ones((vector_shape)), np.eye(vector_shape)
    else:
        U, D, Vt = np.linalg.svd(cov)
    return U, D, Vt

'''
    Given any random sample v with 
      E[ v ] = [0, 0, ..., 0]
      E[ v @ v.T] = Identity
    return correlated samples with covariance specified by cov.
    This can be done by:
      r = cov^{1/2} @ v
      E[ r @ r.T ] = cov
'''
def setCovarianceOfSamples(samples, cov, vector_shape):
    U, D, Vt = covarianceSvd(vector_shape, cov)
    return U @ np.diag(D**0.5) @ Vt @ samples

def scaledIdentityCovGenerator():
    def generator(mean, cov, n_random_vectors):
        vecs = np.random.normal(0, cov[0, 0]**0.5, (mean.shape[0], n_random_vectors)) + mean[:, np.newaxis]
        return vecs.T
    return generator

'''
    l-sparse distribution is defined by 
       s = 1 / (1-l)
                     { +1 with probability 1/(2s)
       v = sqrt(s) * {  0 with probability 1 - 1/s
                     { -1 with probability 1/(2s)
    
    This random variable has the following properties:
      E[v] = 0
      E[v^2] = 1

    By sampling a vector with each component an i.i.d. sample from this distribution, 
    we get a vector with the following properties:
      E[v] = [0, 0, ..., 0]
      E[v @ v.T] = Identity

    Then correlate this sample by calling setCovarianceOfSamples.
'''
def lSparseRandomGenerator(sparsity_level):
    def generator(mean, cov, n_random_vectors):
        # sample from l-percent sparse distribution
        s = 1 / (1 - sparsity_level)
        cutoff = 1/(2*s)
        uniform_samples = np.random.uniform(0, 1, (mean.shape[0], n_random_vectors))
        positive_mask = np.zeros(uniform_samples.shape)
        negative_mask = np.zeros(uniform_samples.shape)
        positive_mask[uniform_samples > 1 - cutoff] = 1.0
        negative_mask[uniform_samples < cutoff] = 1.0
        sparse_sample = (s**0.5) * (positive_mask - negative_mask)

        # transform samples to have correct covariance
        sparse_sample = setCovarianceOfSamples(sparse_sample, cov, mean.shape[0]) + mean[:, np.newaxis]
        return sparse_sample.T
    return generator

'''
    A Rademacher random variable is a special case of an 
    l-percent sparse random variable with l = 1. That is, 
           { +1 with probability 1/2
       v = {  0 with probability 0
           { -1 with probability 1/2
'''
def rademacherRandomGenerator():
    return lSparseRandomGenerator(0)

'''
    An Achlioptas random variable is a special case of an 
    l-percent sparse random variable with l = 2/3.
'''
def achlioptasRandomGenerator():
    return lSparseRandomGenerator(2/3)

'''
    This function generates creates a generator for exponential random variables 
    with mean 0 and covariance specified by cov. 
    This random variable has the following properties:
      E[v] = 1
      E[v^2] = 1

    By sampling a vector with each component an i.i.d. sample from this distribution, 
    we get a vector with the following properties:
      E[v] = [1, 1, ..., 1]
      E[v @ v.T] = Identity

    Subtract 1 from each sample to center them. 

    Then correlate this sample by calling setCovarianceOfSamples.
'''
def exponentialRandomGenerator():
    def generator(mean, cov, n_random_vectors):
        sample = np.random.exponential(size=(mean.shape[0], n_random_vectors))
        sample = sample - 1
        sample = setCovarianceOfSamples(sample, cov, mean.shape[0]) + mean[:, np.newaxis]
        return sample.T
    return generator
