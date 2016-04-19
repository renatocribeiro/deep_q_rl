""" Utility to plot the layers of the just ram network of
the Deep q-network.

Usage:

plot_filters.py PICKLED_NN_FILE
"""

import matplotlib.pyplot as plt
import cPickle
import argparse
import numpy as np
import lasagne.layers

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, nargs=1)
args = parser.parse_args()

net_file = open(args.filename[0], 'r')
network = cPickle.load(net_file)
q_layers = lasagne.layers.get_all_layers(network.l_out)
# TODO: check dimensions

prev = np.identity(q_layers[1].W.get_value().shape[0])

for i in range(1, len(q_layers)):
    act = q_layers[i].W.get_value()
    act = np.dot(prev, act)
    plt.subplot(1, len(q_layers)-1, i)
    plt.imshow(act, vmin=act.min(), vmax=act.max(), interpolation='none',
               cmap='gray')#, aspect=1/32.)
    prev = act

# each column corresponds to the pattern in ram that causes the biggest activation in the given node (assuming that the l2 norm of that ram signal is limited)

plt.tight_layout()
plt.show()
