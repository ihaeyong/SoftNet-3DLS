# Imports
#%matplotlib inline
#%config InlineBackend.figure_formats = ['svg']

import numpy as np  # Matrix and vector computation package
import matplotlib
import matplotlib.pyplot as plt  # Plotting library
from matplotlib import cm # Colormaps
from matplotlib.colors import colorConverter, ListedColormap
import seaborn as sns  # Fancier plots

from sklearn.decomposition import PCA
from utils import *

# Set seaborn plotting style
sns.set_style('darkgrid')
# Set the seed for reproducability
np.random.seed(seed=1)
#

# Define and generate the samples
nb_of_samples_per_class = 20  # The number of sample in each class
red_mean = (-1., 0., 0.)  # The mean of the red class
blue_mean = (1., 0., 0.)  # The mean of the blue class

n_dim = 3
net_type = 'hardnet'

# Generate samples from both classes
x_red = np.random.randn(nb_of_samples_per_class, n_dim) + red_mean
x_blue = np.random.randn(nb_of_samples_per_class, n_dim)  + blue_mean

# Merge samples in set of input variables x, and corresponding 
# set of output variables t
X = np.vstack((x_red, x_blue))
t = np.vstack((np.zeros((nb_of_samples_per_class,1)),
               np.ones((nb_of_samples_per_class,1))))

import ipdb; ipdb.set_trace()
# Plot the loss in function of the weights
# Define a vector of weights for which we want to plot the loss
nb_of_ws = 25 # compute the loss nb_of_ws times in each dimension
wsa = np.linspace(-5, 5, num=nb_of_ws) # weight a
wsb = np.linspace(-5, 5, num=nb_of_ws) # weight b
wsc = np.linspace(-5, 5, num=nb_of_ws) # weight c
ws_x, ws_y, ws_z = np.meshgrid(wsa, wsb, wsc) # generate grid
loss_ws = np.zeros((nb_of_ws, nb_of_ws, nb_of_ws)) # initialize loss matrix

# Fill the loss matrix for each combination of weights
for i in range(nb_of_ws):
    for j in range(nb_of_ws):
        for k in range(nb_of_ws):
            loss_ws[i,j, k] = loss(
                nn(X, np.asmatrix([ws_x[i,j,k], ws_y[i,j,k], ws_z[i,j,k]])) , t)

# Plot the loss function surface
plt.figure(figsize=(6, 4))
plt.contourf(ws_x, ws_y, loss_ws, 20, cmap=cm.viridis)
cbar = plt.colorbar()
cbar.ax.set_ylabel('$\\xi$', fontsize=12)
plt.xlabel('$w_1$', fontsize=12)
plt.ylabel('$w_2$', fontsize=12)
plt.title('Loss function surface')
plt.grid()
plt.savefig('./plots/{}_loss_func_surface.pdf'.format(net_type),
            dpi=300, format='pdf', bbox_inches='tight')
plt.close()


