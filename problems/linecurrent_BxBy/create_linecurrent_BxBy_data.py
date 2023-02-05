#!/usr/bin/env python

"""Compute initial conditions for linecurrent_BxBy.

Author
------
eric.winter62@gmail.com
"""


import sys

import numpy as np


# Constants
mu0 = 1.0
I = 1.0
Q = 60.0
u0 = 1.0
nt = 50
nx = 50
ny = 50

# Compute the domain limits.
t_min = 0.0
t_max = 1.0
x_min = -1.0
x_max = 1.0
y_min = -1.0
y_max =  1.0
# print("t_min, t_max = %s, %s" % (t_min, t_max))
# print("x_min, x_max = %s, %s" % (x_min, x_max))
# print("y_min, y_max = %s, %s" % (y_min, y_max))

# Compute the grid locations in each dimension.
tg = np.linspace(t_min, t_max, nt)
xg = np.linspace(x_min, x_max, nx)
yg = np.linspace(y_min, y_max, ny)
# print("tg = %s" % tg)
# print("xg = %s" % xg)
# print("yg = %s" % yg)

# Compute the constant velocity components.
ux = u0*np.sin(np.radians(Q))
uy = u0*np.cos(np.radians(Q))
# print("ux, uy = %s, %s" % (ux, uy))

# Compute the initial conditions at spatial grid locations.
# Each line is:
# t x y Bx By
for (i, x) in enumerate(xg):
    for (j, y) in enumerate(yg):
        r = np.sqrt(x**2 + y**2)
        Bx = -mu0*I/(2*np.pi)*y/r**2
        By = mu0*I/(2*np.pi)*x/r**2
        print(tg[0], x, y, Bx, By)
