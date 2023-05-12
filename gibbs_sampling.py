import numpy as np
import warnings
from scipy import stats
import os
import multiprocessing
from itertools import repeat
import matplotlib.pyplot as plt

#suppress warnings
warnings.filterwarnings('ignore')

# Auxilliary function for the sigmoid of a vector term-wise
def sigmoid(input):
    return [1/(1+np.exp(-v)) for v in input]


# Sample the moment parameters of the model
def sample_parameters(model_dims):
    m, n = model_dims[0], model_dims[1]
    p = np.random.uniform(0, 1, m)
    mu = np.random.uniform(-100, 100, n)
    sigma = np.random.uniform(0, 100, n)
    W = np.random.uniform(-100,100, (n, m))
    return [p, mu, sigma, W]


# Function to initialize the Gibbs sampling
def initialize_GRBM(model_dims, params):
    latent = np.random.randint(1.1, size=model_dims[0])
    observed_aux = np.random.rand(model_dims[1])
    observed = params[1] + params[3].dot(latent)
    return np.concatenate((latent, observed))

    
# Single sample generator, works in four steps; calculates latent parameters, samples latent,
# calculates observable parameters, samples observables; returns list [m, n] sample
def one_step_sample(model_dims, current_state, params):
    x = current_state[model_dims[0]:]
    Wt = params[3].transpose() # shape (m, n)
    ps = sigmoid(Wt.dot(np.multiply(x, 1/params[2])) + params[0])
    latent = [np.random.binomial(1, p) for p in ps]
    mu = params[1] + params[3].dot(latent)
    S = np.diag(params[2])
    observed = np.random.multivariate_normal(mu, S)
    return np.concatenate((latent, observed))


# model_dims is a two-dimensional vector with the number of latent, observables
# initial_value is an initial sample 
# This function samples from a GRBM model using a Gibbs Sampling method,
# explioting the independence structure of a RBM Graphical Model
def GRBM_gibbs_sampling(model_dims, params, initial_value, number_of_samples):
    samples = np.zeros((number_of_samples, model_dims[1]))
    previous_sample = initial_value
    for i in range(number_of_samples):
        sample = one_step_sample(model_dims, previous_sample, params)
        samples[i] = sample[model_dims[0]:]
        previous_sample = sample
    return samples


# Function to pack the procedure of obtaining the sample correlation matrix
def generate_sample_correlation(model_dims, gibbs_samples, Rhos):
    params = sample_parameters(model_dims)
    initial_value = initialize_GRBM(model_dims, params)
    samples = GRBM_gibbs_sampling(model_dims, params, initial_value, gibbs_samples)
    Rho = np.corrcoef(samples.transpose()) # Correlation matrix, note this is very expensive
    return Rho


#### MAIN

np.random.seed(0)
total_samples = 18000 # Number of correlation triplets produced
gibbs_samples = 500 # Number of samples to compute correlation from

model_dims = [100, 3]

Rhos = np.zeros((total_samples, model_dims[1]))


if __name__ == '__main__':
    # Parallelism to speed up the samples
    with multiprocessing.Pool(os.cpu_count()) as pool:
        for i, Rho in enumerate(pool.starmap(generate_sample_correlation, zip(repeat(model_dims), repeat(gibbs_samples), Rhos), chunksize=int(total_samples/os.cpu_count()))):
            Rhos[i] = [Rho[0,1], Rho[0,2], Rho[1,2]]

    results = Rhos

    #Plot of observations
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(results[:,0], results[:,1], results[:,2], c=1-results[:,0]-results[:,1]-results[:,2], cmap='Reds')
    plt.show()
"""
    fig2, axs = plt.subplots(3)
    fig2.suptitle('Hisotgram subplots')
    n_bins = 30
    ax0, ax1, ax2 = axs.flatten()

    ax0.hist(samples[:,0], n_bins, histtype='bar')
    ax0.set_title('X1')

    ax1.hist(samples[:,1], n_bins, histtype='bar')
    ax1.set_title('X2')

    ax2.hist(samples[:,2], n_bins, histtype='bar')
    ax2.set_title('X3')
"""
    

