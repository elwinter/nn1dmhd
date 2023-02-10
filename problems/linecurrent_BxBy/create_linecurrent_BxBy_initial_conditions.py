#!/usr/bin/env python

"""Compute initial conditions for linecurrent_BxBy.

Author
------
eric.winter62@gmail.com
"""


# Import standard Python modules.
import argparse

# Import supplemental Python modules.
import numpy as np

# Import project Python modules.


# Program constants

# Program description.
description = "Compute data for linecurrent_BxBy problem."

# Default random number generator seed.
default_seed = 0

# Constants
mu0 = 1.0  # Normalized vacuum permittivity.
I = 1.0    # Normalized current.
Q = 60.0   # Flow angle in degrees clockwise from +y axis.
u0 = 1.0   # Initial flow speed.

# Compute the constant velocity components.
ux = u0*np.sin(np.radians(Q))
uy = u0*np.cos(np.radians(Q))


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
    # They should be in 3 sets of 3:
    # t_min t_max n_t x_min x_max n_x y_min y_max n_y
    assert len(rest) == 9
    X_min = np.array(rest[::3], dtype=float)
    X_max = np.array(rest[1::3], dtype=float)
    X_n = np.array(rest[2::3], dtype=int)
    if debug:
        print("X_min = %s" % X_min)
        print("X_max = %s" % X_max)
        print("X_n = %s" % X_n)
    assert len(X_min) == len(X_max) == len(X_n)

    # Extract limits for convenience.
    (t_min, x_min, y_min) = X_min
    (t_max, x_max, y_max) = X_max
    (n_t, n_x, n_y) = X_n
    if debug:
        print("%s <= t <= %s" % (t_min, t_max))
        print("%s <= x <= %s" % (x_min, x_max))
        print("%s <= y <= %s" % (y_min, y_max))
    
    # Create the (x, y) coordinate points for the initial conditions.
    # Points are either random or gridded.
    if random:
        np.random.seed(seed)
        xg = x_min + np.random.random_sample((n_x,))*(x_max - x_min)
        yg = y_min + np.random.random_sample((n_y,))*(y_max - y_min)
    else:
        xg = np.linspace(x_min, x_max, n_x)
        yg = np.linspace(y_min, y_max, n_y)
    if debug:
        print("xg = %s" % xg)
        print("yg = %s" % yg)

    # Compute the initial conditions at spatial locations.
    # Each line is:
    # t_min x y Bx By
    for (i, x) in enumerate(xg):
        for (j, y) in enumerate(yg):
            r = np.sqrt(x**2 + y**2)
            Bx = -mu0*I/(2*np.pi)*y/r**2
            By = mu0*I/(2*np.pi)*x/r**2
            print(t_min, x, y, Bx, By)


if __name__ == "__main__":
    """Begin main program."""
    main()
