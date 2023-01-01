#!/usr/bin/env python

# Plot a 2D set of training and data points.


# Import standard modules.
import sys

# Import supplemental modules.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# Specify training points file.
training_points_path = sys.argv[1]

# Specify data points file.
data_points_path = sys.argv[2]

# Read the training points into a 2-D array.
training_points = np.loadtxt(training_points_path)
print(training_points)

# Read the data points.
data_points = np.loadtxt(data_points_path)
# If only 1 point, put it in a 2-D array of 1 row, 2 + n_var columns.
if len(data_points.shape) == 1:
    data_points = data_points.reshape(1, -1)
print(data_points)

# Extract only the coordinates of the training data.
data_points = data_points[:, :2]
print(data_points)

# Create plots in-memory.
mpl.use("AGG")

# Plot the training points on a line, then the data points.
plt.figure()
plt.scatter(training_points[:, 0], training_points[:, 1], s=60, c='r')
plt.scatter(data_points[:, 0], data_points[:, 1], s=20, c='g')
plt.grid()
plt.savefig("train2D.png")