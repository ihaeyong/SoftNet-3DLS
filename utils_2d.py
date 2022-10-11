# Imports
#%matplotlib inline
#%config InlineBackend.figure_formats = ['svg']

import numpy as np  # Matrix and vector computation package
import matplotlib
import matplotlib.pyplot as plt  # Plotting library
from matplotlib import cm # Colormaps
from matplotlib.colors import colorConverter, ListedColormap
import seaborn as sns  # Fancier plots

# Define the logistic function
def logistic(z):
    return 1. / (1 + np.exp(-z))

# Percentile
def percentile(scores, sparsity):
    k = 1 + round(.01 + float(sparsity) * (len(scores.flatten()) - 1))
    return sorted(np.array(scores).flatten())[-k]

# Define the HardNet
def hardnet(w, sparsity):
    k_val = percentile(np.abs(w), sparsity*100.0)
    m_hard = np.where(np.abs(w) < k_val, 0, 1)
    return m_hard

# Define the SoftNet
#def softnet(w, sparsity):
#	k_val = percentile(w.abs(), sparsity*100.0)
#	ones = np.ones_like(w.abs())
#    rand = np.rand_like(w.abs()
#    m_soft = np.where(w.abs() < k_val, rand, ones)
#    return m_soft

# Define the neural network function y = 1 / (1 + numpy.exp(-x*w))
def nn(x, w, n_type='hardnet'):
    if n_type == 'hardnet':
        m=hardnet(w, sparsity=0.1)
        w=np.multiply(w,m)
    return logistic(x.dot(w.T))

# Define the neural network function y = 1 / (1 + np.exp(-x * w))
def nn_prune(x, w, n_type='dense', sparisty=0.1):
	if n_type == 'hardnet':
		m = hardnet(w, sparsity=0.1)
		w = w * m
	elif n_type == 'softnet':
		m = softnet(w, sparsity=0.1)
		w = w * m
	else:
		None
	return logistic(x.dot(w.T))


# Define the neural network prediction function that only returns
#  1 or 0 depending on the predicted class
def nn_predict(x,w):
    return np.around(nn(x,w))

# Define the loss function
def loss(y, t):
    return - np.mean(
        np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))

def gradient(w, x, t):
    """Gradient function."""
    return (nn(x, w) - t).T * x

def delta_w(w_k, x, t, learning_rate):
    """Update function which returns the update for each
    weight {w_a, w_b} in a vector."""
    return learning_rate * gradient(w_k, x, t)


