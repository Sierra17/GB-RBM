import numpy as np

# vector of numeric entries
def sigmoid(X):
    return [1/(1+np.exp(-x)) for x in X]

class RBM:

  def __init__(self, num_latent, num_observable):
    self.num_latent = num_latent
    self.num_observable = num_observable

  def train(self, data, max_epochs=10, learning_rate=0.1):
    for epoch in range(max_epochs):
      
class GBRBM:

  def __init__(self, num_latent, num_observable):
    self.num_latent = num_latent
    self.num_observable = num_observable
