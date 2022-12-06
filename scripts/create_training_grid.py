#!/usr/bin/env python

"""Create a training grid.

This program will create a grid of training points.

THe user supplies the minimum and maximum values for each dimension, and
the number of points in each dimension.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard Python modules.
import sys

# Import 3rd-party modules.
import numpy as np

# Import project modules.
from nn1dmhd.training_data import create_training_points_gridded


# Fetch all of the command-line arguments.
# They should be in sets of 3:
# x_min x_max n_x y_min y_max n_y ...
args = sys.argv[1:]
assert(len(args) % 3 == 0)

# Fetch the minimum for each dimension.
X_min = np.array([float(x) for x in args[::3]])

# Fetch the maximum for each dimension.
X_max = np.array([float(x) for x in args[1::3]])

# Fetch the point count for each dimension.
ng = np.array([int(x) for x in args[2::3]])

# Assemble the minima and maxima into a combined array of the form:
# [
#  [x_min, x_max],
#  [y_min, y_max],
#  ...
# ]
bg = np.vstack([X_min, X_max]).T

# Create the flattened, evenly-spaced grid. The last dimension varies fastest.
grid = create_training_points_gridded(ng, bg)

# Save the grid to a file.s
np.savetxt("grid.dat", grid, fmt="%g")

# To determine the number of grid points from the output file:
# n_x = len(np.unique(grid[:, 0])))
# n_y = len(np.unique(grid[:, 1])))
# etc.