#!/usr/bin/env python

# Plot a 1D set of training and data points.


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
training_points.shape = (len(training_points), 1)
# print(training_points)

# Read the data points.
data_points = np.loadtxt(data_points_path)
# If only 1 point, put it in a 2-D array of 1 row, 1 column.
if len(data_points.shape) == 1:
    data_points = np.array([[data_points[0]]]).reshape(1, 1)
# print(data_points)

# Extract only the coordinates of the training data.
data_points = data_points[:, 0].reshape(len(data_points))
# print(data_points)

# Create plots in-memory.
# mpl.use("AGG")
# print(training_points[:, 0])
# print(data_points[:, 0])

# Compute plot limits.
xtmin = np.min(training_points.flatten())
xdmin = np.min(data_points.flatten())
xmin = np.min([xtmin, xdmin])
xtmax = np.max(training_points.flatten())
xdmax = np.max(data_points.flatten())
xmax = np.max([xtmax, xdmax])

# Plot the training points on a line, then the data points.
plt.figure()
plt.hlines(0, xmin, xmax)
plt.scatter(training_points, [0]*len(training_points), s=60, c='r')
plt.scatter(data_points, [0]*len(data_points), s=20, c='g')
plt.grid()
plt.savefig("train1D.png")