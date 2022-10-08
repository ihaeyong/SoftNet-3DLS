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

# Define the neural network function y = 1 / (1 + numpy.exp(-x*w))
def nn(x, w):
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


