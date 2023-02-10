"""Problem definition file for a simple 2-D MHD problem.

This problem definition file describes the 2-D line current convection
problem, which is based on the loop2d example in the Athena MHD test suite.
Details are available at:

https://www.astro.princeton.edu/~jstone/Athena/tests/field-loop/Field-loop.html

NOTE: This case deals only with a line current in the +z direction (out of
the screen). +x is to the right, +y is up.

NOTE: This version of the code solves *only* the equations for n, P, ux, uy,
Bx, By, and Bz.

NOTE: The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by the TensorFlow
code.

NOTE: In all code, below, the following indices are assigned to physical
independent variables:

    0: t
    1: x
    2: y

NOTE: In all code, below, the following indices are assigned to physical
dependent variables:

    0: n (number density)
    1: P (pressure)
    2: ux (x-component of velocity)
    3: uy (y-component of velocity)
    4: Bx (x-component of magnetic field)
    5: By (y-component of magnetic field)
    6: Bz (z-component of magnetic field)

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard modules.

# Import supplemental modules.
import numpy as np
import tensorflow as tf

# Import project modules.


# Names of independent variables.
independent_variable_names = ['t', 'x', 'y']

# Labels for independent variables (may use LaTex) - use for plots.
independent_variable_labels = ["$t$", "$x$", "$y$"]

# Number of problem dimensions (independent variables).
n_dim = len(independent_variable_names)

# Names of dependent variables.
dependent_variable_names = ['n', 'P', 'ux', 'uy', 'Bx', 'By', 'Bz']

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = [
    "$n$", "$P$",
    "$u_x$", "$u_y$",
    "$B_x$", "$B_y$", "$B_z$"
]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# Normalized physical constants.
μ0 = 1.0  # Permeability of free space
m = 1.0   # Particle mass
ɣ = 5/3   # Adiabatic index = (N + 2)/N, N = # DOF=3, not 2.


# NOTE: In the functions defined below for the differential equations, the
# arguments can be unpacked as follows:
# def pde_XXX(X, Y, del_Y):
#     nX = X.shape[0]
#     t = tf.reshape(X[:, 0], (nX, 1))
#     x = tf.reshape(X[:, 1], (nX, 1))
#     y = tf.reshape(X[:, 2], (nX, 1))
#     (n, P, ux, uy, Bx, By, Bz) = Y
#     (del_n, del_P, del_ux, del_uy, del_Bx, del_By, del_Bz) = del_Y
#     dn_dt = tf.reshape(del_n[:, 0], (nX, 1))
#     dn_dx = tf.reshape(del_n[:, 1], (nX, 1))
#     dn_dy = tf.reshape(del_n[:, 2], (nX, 1))
#     dP_dt = tf.reshape(del_P[:, 0], (nX, 1))
#     dP_dx = tf.reshape(del_P[:, 1], (nX, 1))
#     dP_dy = tf.reshape(del_P[:, 2], (nX, 1))
#     dux_dt = tf.reshape(del_ux[:, 0], (nX, 1))
#     dux_dx = tf.reshape(del_ux[:, 1], (nX, 1))
#     dux_dy = tf.reshape(del_ux[:, 2], (nX, 1))
#     duy_dt = tf.reshape(del_uy[:, 0], (nX, 1))
#     duy_dx = tf.reshape(del_uy[:, 1], (nX, 1))
#     duy_dy = tf.reshape(del_uy[:, 2], (nX, 1))
#     dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
#     dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
#     dBx_dy = tf.reshape(del_Bx[:, 2], (nX, 1))
#     dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
#     dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
#     dBy_dy = tf.reshape(del_By[:, 2], (nX, 1))
#     dBz_dt = tf.reshape(del_Bz[:, 0], (nX, 1))
#     dBz_dx = tf.reshape(del_Bz[:, 1], (nX, 1))
#     dBz_dy = tf.reshape(del_Bz[:, 2], (nX, 1))


# @tf.function
def pde_n(X, Y, del_Y):
    """Differential equation for number density.

    Evaluate the differential equation for number density. This equation is
    derived from the equation of mass continuity.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    del_Y : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables
        at each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.
    """
    nX = X.shape[0]
    # t = tf.reshape(X[:, 0], (nX, 1))
    # x = tf.reshape(X[:, 1], (nX, 1))
    # y = tf.reshape(X[:, 2], (nX, 1))
    (n, P, ux, uy, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_Bx, del_By, del_Bz) = del_Y
    dn_dt = tf.reshape(del_n[:, 0], (nX, 1))
    dn_dx = tf.reshape(del_n[:, 1], (nX, 1))
    dn_dy = tf.reshape(del_n[:, 2], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, 0], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, 1], (nX, 1))
    # dP_dy = tf.reshape(del_P[:, 2], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, 0], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, 1], (nX, 1))
    # dux_dy = tf.reshape(del_ux[:, 2], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, 0], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, 1], (nX, 1))
    duy_dy = tf.reshape(del_uy[:, 2], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
    # dBx_dy = tf.reshape(del_Bx[:, 2], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, 2], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 0], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, 1], (nX, 1))
    # dBz_dy = tf.reshape(del_Bz[:, 2], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dn_dt + n*(dux_dx + duy_dy) + dn_dx*ux + dn_dy*uy
    return G


# @tf.function
def pde_P(X, Y, del_Y):
    """Differential equation for P.

    Evaluate the differential equation for pressure (or energy density). This
    equation is derived from the equation of conservation of energy.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    del_Y : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables
        at each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.
    """
    nX = X.shape[0]
    # t = tf.reshape(X[:, 0], (nX, 1))
    # x = tf.reshape(X[:, 1], (nX, 1))
    # y = tf.reshape(X[:, 2], (nX, 1))
    (n, P, ux, uy, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_Bx, del_By, del_Bz) = del_Y
    dn_dt = tf.reshape(del_n[:, 0], (nX, 1))
    dn_dx = tf.reshape(del_n[:, 1], (nX, 1))
    dn_dy = tf.reshape(del_n[:, 2], (nX, 1))
    dP_dt = tf.reshape(del_P[:, 0], (nX, 1))
    dP_dx = tf.reshape(del_P[:, 1], (nX, 1))
    dP_dy = tf.reshape(del_P[:, 2], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, 0], (nX, 1))
    # dux_dx = tf.reshape(del_ux[:, 1], (nX, 1))
    # dux_dy = tf.reshape(del_ux[:, 2], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, 0], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, 1], (nX, 1))
    # duy_dy = tf.reshape(del_uy[:, 2], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
    # dBx_dy = tf.reshape(del_Bx[:, 2], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, 2], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 0], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, 1], (nX, 1))
    # dBz_dy = tf.reshape(del_Bz[:, 2], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = -ɣ*P/n*(dn_dt + ux*dn_dx + uy*dn_dy) + (dP_dt + ux*dP_dx + uy*dP_dy)/m
    return G


# @tf.function
def pde_ux(X, Y, del_Y):
    """Differential equation for x-velocity.

    Evaluate the differential equation for x-velocity. This equation is derived
    from the equation of conservation of x-momentum.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    del_Y : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables at
        each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.
    """
    nX = X.shape[0]
    # t = tf.reshape(X[:, 0], (nX, 1))
    # x = tf.reshape(X[:, 1], (nX, 1))
    # y = tf.reshape(X[:, 2], (nX, 1))
    (n, P, ux, uy, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_Bx, del_By, del_Bz) = del_Y
    # dn_dt = tf.reshape(del_n[:, 0], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, 1], (nX, 1))
    # dn_dy = tf.reshape(del_n[:, 2], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, 0], (nX, 1))
    dP_dx = tf.reshape(del_P[:, 1], (nX, 1))
    # dP_dy = tf.reshape(del_P[:, 2], (nX, 1))
    dux_dt = tf.reshape(del_ux[:, 0], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, 1], (nX, 1))
    dux_dy = tf.reshape(del_ux[:, 2], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, 0], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, 1], (nX, 1))
    # duy_dy = tf.reshape(del_uy[:, 2], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
    dBx_dy = tf.reshape(del_Bx[:, 2], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
    dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, 2], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 0], (nX, 1))
    dBz_dx = tf.reshape(del_Bz[:, 1], (nX, 1))
    # dBz_dy = tf.reshape(del_Bz[:, 2], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        n*(dux_dt + ux*dux_dx + uy*dux_dy) + dP_dx/m +
        (By*(dBy_dx - dBx_dy) + Bz*dBz_dx)/(m*μ0)
    )
    return G


# @tf.function
def pde_uy(X, Y, del_Y):
    """Differential equation for y-velocity.

    Evaluate the differential equation for y-velocity. This equation is derived
    from the equation of conservation of y-momentum.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    del_Y : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables at
        each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.
    """
    nX = X.shape[0]
    # t = tf.reshape(X[:, 0], (nX, 1))
    # x = tf.reshape(X[:, 1], (nX, 1))
    # y = tf.reshape(X[:, 2], (nX, 1))
    (n, P, ux, uy, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_Bx, del_By, del_Bz) = del_Y
    # dn_dt = tf.reshape(del_n[:, 0], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, 1], (nX, 1))
    # dn_dy = tf.reshape(del_n[:, 2], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, 0], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, 1], (nX, 1))
    dP_dy = tf.reshape(del_P[:, 2], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, 0], (nX, 1))
    # dux_dx = tf.reshape(del_ux[:, 1], (nX, 1))
    # dux_dy = tf.reshape(del_ux[:, 2], (nX, 1))
    duy_dt = tf.reshape(del_uy[:, 0], (nX, 1))
    duy_dx = tf.reshape(del_uy[:, 1], (nX, 1))
    duy_dy = tf.reshape(del_uy[:, 2], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
    dBx_dy = tf.reshape(del_Bx[:, 2], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
    dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, 2], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 0], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, 1], (nX, 1))
    dBz_dy = tf.reshape(del_Bz[:, 2], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        n*(duy_dt + ux*duy_dx + uy*duy_dy) + dP_dy/m +
        (Bx*(dBx_dy - dBy_dx) + Bz*dBz_dy)/(m*μ0)
    )
    return G


# @tf.function
def pde_Bx(X, Y, del_Y):
    """Differential equation for x-magnetic field.

    Evaluate the differential equation for x-magnetic field. This equation is
    derived from the x-component of Faraday's Law.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    del_Y : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables at
        each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.
    """
    nX = X.shape[0]
    # t = tf.reshape(X[:, 0], (nX, 1))
    # x = tf.reshape(X[:, 1], (nX, 1))
    # y = tf.reshape(X[:, 2], (nX, 1))
    (n, P, ux, uy, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_Bx, del_By, del_Bz) = del_Y
    # dn_dt = tf.reshape(del_n[:, 0], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, 1], (nX, 1))
    # dn_dy = tf.reshape(del_n[:, 2], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, 0], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, 1], (nX, 1))
    # dP_dy = tf.reshape(del_P[:, 2], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, 0], (nX, 1))
    # dux_dx = tf.reshape(del_ux[:, 1], (nX, 1))
    dux_dy = tf.reshape(del_ux[:, 2], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, 0], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, 1], (nX, 1))
    duy_dy = tf.reshape(del_uy[:, 2], (nX, 1))
    dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
    dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
    dBx_dy = tf.reshape(del_Bx[:, 2], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, 2], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 0], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, 1], (nX, 1))
    # dBz_dy = tf.reshape(del_Bz[:, 2], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dBx_dt + ux*dBx_dx + uy*dBx_dy + Bx*duy_dy - By*dux_dy
    return G


# @tf.function
def pde_By(X, Y, del_Y):
    """Differential equation for y-magnetic field.

    Evaluate the differential equation for y-magnetic field. This equation is
    derived from the y-component of Faraday's Law.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    del_Y : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables at
        each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.
    """
    nX = X.shape[0]
    # t = tf.reshape(X[:, 0], (nX, 1))
    # x = tf.reshape(X[:, 1], (nX, 1))
    # y = tf.reshape(X[:, 2], (nX, 1))
    (n, P, ux, uy, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_Bx, del_By, del_Bz) = del_Y
    # dn_dt = tf.reshape(del_n[:, 0], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, 1], (nX, 1))
    # dn_dy = tf.reshape(del_n[:, 2], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, 0], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, 1], (nX, 1))
    # dP_dy = tf.reshape(del_P[:, 2], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, 0], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, 1], (nX, 1))
    # dux_dy = tf.reshape(del_ux[:, 2], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, 0], (nX, 1))
    duy_dx = tf.reshape(del_uy[:, 1], (nX, 1))
    # duy_dy = tf.reshape(del_uy[:, 2], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
    # dBx_dy = tf.reshape(del_Bx[:, 2], (nX, 1))
    dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
    dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
    dBy_dy = tf.reshape(del_By[:, 2], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 0], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, 1], (nX, 1))
    # dBz_dy = tf.reshape(del_Bz[:, 2], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dBy_dt + ux*dBy_dx + uy*dBy_dy + By*dux_dx - Bx*duy_dx
    return G


# @tf.function
def pde_Bz(X, Y, del_Y):
    """Differential equation for z-magnetic field.

    Evaluate the differential equation for z-magnetic field. This equation is
    derived from the z-component of Faraday's Law.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    del_Y : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables at
        each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.
    """
    nX = X.shape[0]
    # t = tf.reshape(X[:, 0], (nX, 1))
    # x = tf.reshape(X[:, 1], (nX, 1))
    # y = tf.reshape(X[:, 2], (nX, 1))
    (n, P, ux, uy, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_Bx, del_By, del_Bz) = del_Y
    # dn_dt = tf.reshape(del_n[:, 0], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, 1], (nX, 1))
    # dn_dy = tf.reshape(del_n[:, 2], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, 0], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, 1], (nX, 1))
    # dP_dy = tf.reshape(del_P[:, 2], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, 0], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, 1], (nX, 1))
    # dux_dy = tf.reshape(del_ux[:, 2], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, 0], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, 1], (nX, 1))
    duy_dy = tf.reshape(del_uy[:, 2], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
    # dBx_dy = tf.reshape(del_Bx[:, 2], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, 2], (nX, 1))
    dBz_dt = tf.reshape(del_Bz[:, 0], (nX, 1))
    dBz_dx = tf.reshape(del_Bz[:, 1], (nX, 1))
    dBz_dy = tf.reshape(del_Bz[:, 2], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dBz_dt + ux*dBz_dx + uy*dBz_dy + Bz*(dux_dx + duy_dy)
    return G


# Make a list of all of the differential equations.
de = [
    pde_n,
    pde_P,
    pde_ux,
    pde_uy,
    pde_Bx,
    pde_By,
    pde_Bz,
]


if __name__ == "__main__":
    print("independent_variable_names = %s" % independent_variable_names)
    print("independent_variable_labels = %s" % independent_variable_labels)
    print("n_dim = %s" % n_dim)
    print("dependent_variable_names = %s" % dependent_variable_names)
    print("dependent_variable_labels = %s" % dependent_variable_labels)
    print("n_var = %s" % n_var)

    print("μ0 = %s" % μ0)
    print("m = %s" % m)
    print("ɣ = %s" % ɣ)
