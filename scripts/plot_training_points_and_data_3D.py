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

# Read the data points.
data_points = np.loadtxt(data_points_path)
# If only 1 point, put it in a 2-D array of 1 row, 2 + n_var columns.
if len(data_points.shape) == 1:
    data_points = data_points.reshape(1, -1)

# Extract only the coordinates of the training data.
data_points = data_points[:, :3]

# Create plots in-memory.
mpl.use("AGG")

# Plot the training points on a line, then the data points.
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(training_points[:, 0], training_points[:, 1], training_points[:, 2], s=30, c='r')
ax.scatter(data_points[:, 0], data_points[:, 1], data_points[:, 2], s=10, c='g')
ax.grid()
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('y')
plt.savefig("train3D.png")
