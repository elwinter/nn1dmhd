#!/usr/bin/env python

"""Create a set of boundary points with values.

This program will create a set of boundary points and associated boundary
values. The set can be an evenly-spaced grid in each dimension (the default),
or random points in each dimension.

The user supplies the minimum and maximum values for each dimension, and
the number of points in each dimension.

All boundary points are assigned the same values.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard Python modules.
import argparse
import sys

# Import 3rd-party modules.
import numpy as np

# Import project modules.
from nn1dmhd.training_data import (
    create_training_points_gridded, create_training_points_random
)


# Program constants

# Program description.
description = "Create a set of training points."

# Default random number generator seed.
default_seed = 0


def create_command_line_argument_parser():
    """Create the command-line argument parser.

    Create the command-line argument parser.

    Parameters
    ----------
    None

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser for command-line arguments.
    """
    parser = argparse.ArgumentParser(description)
    parser.add_argument(
        "-d", "--debug", action="store_true",
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "-r", "--random", action="store_true",
        help="Select points randomly within domain (default: %(default)s)."
    )
    parser.add_argument(
        "--seed", type=int, default=default_seed,
        help="Seed for random number generator (default: %(default)s)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    return parser


def main():
    """Begin main program."""

    # Set up the command-line parser.
    parser = create_command_line_argument_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    debug = args.debug
    random = args.random
    seed = args.seed
    verbose = args.verbose
    rest = args.rest
    if debug:
        print("args = %s" % args)

    # Fetch the remaining command-line arguments.
    # They should be in sets of 3, plus 1:
    # x_min x_max n_x y_min y_max n_y d0,d1,d2,...
    X_min = np.array(rest[:-1:3], dtype=float)
    X_max = np.array(rest[1:-1:3], dtype=float)
    X_n = np.array(rest[2:-1:3], dtype=int)
    bc_str = rest[-1]
    if debug:
        print("X_min = %s" % X_min)
        print("X_max = %s" % X_max)
        print("X_n = %s" % X_n)
        print("bc_str = %s" % bc_str)
    assert len(X_min) == len(X_max) == len(X_n)

    # Assemble the minima and maxima into a combined array of boundaries of
    # the form:
    # [
    #  [x_min, x_max],
    #  [y_min, y_max],
    #  ...
    # ]
    b = np.vstack([X_min, X_max]).T
    if debug:
        print("b = %s" % b)

    # Create the training points.
    if random:

        # Seed the random number generator.
        np.random.seed(seed)

        # Select the training points randomly within the domain.
        points = create_training_points_random(X_n, b)

    else:
        # Create the flattened, evenly-spaced grid. The last dimension varies
        # fastest.
        points = create_training_points_gridded(X_n, b)

    # Convert the data string to a list. Split on spaces.
    bc = [float(d) for d in bc_str.split()]

    # Now that the points are created, determine which points lie on
    # boundaries and assign points to them. Discard non-boundary points.
    data = []
    for X in points:
        for (i, x) in enumerate(list(X)):
            if np.isclose(x, b[i, 0]) or np.isclose(x, b[i, 1]):
                data.append(list(X) + bc)
                break

    # Send the points and data to standard output.
    np.savetxt(sys.stdout, np.array(data), fmt="%g")

    # NOTE:
    # To determine the number of *grid* points from the output file:
    # n_x = len(np.unique(grid[:, 0])))
    # n_y = len(np.unique(grid[:, 1])))
    # etc.


if __name__ == "__main__":
    """Begin main program."""
    main()
