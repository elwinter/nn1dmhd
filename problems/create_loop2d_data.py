#!/usr/bin/env python

"""Compute initial conditions for loop2D.

Author
------
eric.winter62@gmail.com
"""


import sys

import numpy as np


# Constants
Q = 60.0
gamma = 5/3
rho0 = 1.0
n0 = 1.0
P0 = 1.0
A = 1e-3
R0 = 0.3
u0 = 1.0
Bz0 = 0.0
uz0 = 0.0
nt = 100
nx = 100
ny = 100

# Compute the domain limits.
t_min = 0.0
t_max = 1.0
x_min = -1.0
x_max = 1.0
y_min = -1/(2*np.cos(np.radians(90 - Q)))
y_max =  1/(2*np.cos(np.radians(90 - Q)))
# print("x_min, x_max = %s, %s" % (x_min, x_max))
# print("y_min, y_max = %s, %s" % (y_min, y_max))

# Compute the grid locations in each dimension.
tg = np.linspace(t_min, t_max, nt)
xg = np.linspace(x_min, x_max, nx)
yg = np.linspace(y_min, y_max, ny)

# Compute the initial conditions at spatial grid locations.
# Each line is:
# t x y n P ux uy uz Bx By Bz
for (i, x) in enumerate(xg):
    for (j, y) in enumerate(yg):
        n = n0
        P = P0
        ux = u0*np.sin(np.radians(Q))
        uy = u0*np.cos(np.radians(Q))
        uz = uz0
        r = np.sqrt(x**2 + y**2)
        if r < R0:
            Bx = -A*y/r
            By =  A*x/r
        else:
            Bx = 0.0
            By = 0.0
        Bz = Bz0
        print(tg[0], x, y, n, P, ux, uy, uz, Bx, By, Bz)
