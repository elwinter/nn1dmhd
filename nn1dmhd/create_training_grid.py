#!/usr/bin/env python

"""Create an evenly-spaced training grid.

This program will create an evenly-spaced training grid in an arbitrary
number of dimensions.

The domain of each dimension is [0, 1].

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
from training_data import create_training_points_gridded


# Program constants

# Program description.
description = "Create a training grid."


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
        "-v", "--verbose", action="store_true",
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument("grid_counts", nargs=argparse.REMAINDER)
    return parser


def main():
    """Begin main program."""

    # Set up the command-line parser.
    parser = create_command_line_argument_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    debug = args.debug
    verbose = args.verbose
    if debug:
        print("args = %s" % args)

    # Convert remaining arguments to integers.
    ns = [int(n) for n in args.grid_counts]

    # Determine the number of dimensions.
    n_dim = len(ns)

    # Create the array of normalized limits.
    limits = [(0, 1) for i in range(n_dim)]

    # Create the normalized training grid.
    grid = np.array(create_training_points_gridded(ns, limits))
    np.savetxt(sys.stdout, grid, fmt="%g")


if __name__ == "__main__":
    """Begin main program."""
    main()
