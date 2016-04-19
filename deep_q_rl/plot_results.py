"""Plots data corresponding to Figure 2 in

Playing Atari with Deep Reinforcement Learning
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis
Antonoglou, Daan Wierstra, Martin Riedmiller

Usage:

plot_results.py RESULTS_CSV_FILE (--epochs E)
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, nargs=1)
parser.add_argument('--epochs', dest='epochs', default=100, type=int, nargs='?')
parser.add_argument('--title', dest='title', default="", type=str, nargs='?')
args = parser.parse_args()

# Modify this to do some smoothing...
kernel = np.array([1.] * 1)
kernel = kernel / np.sum(kernel)

results = np.loadtxt(open(args.filename[0], "rb"), delimiter=",", skiprows=1)
plt.plot(results[:, 0], np.convolve(results[:, 3], kernel, mode='same'), '-')
plt.xlabel('Training Epochs')
plt.ylabel('Average score per episode')
plt.title(args.title)
plt.xlim([0, args.epochs])
plt.savefig("figure.png")
