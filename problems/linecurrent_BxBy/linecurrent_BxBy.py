"""Problem definition file for a simple 2-D MHD problem.

This problem definition file describes the 2-D line current advection
problem, which is based on the loop2d example in the Athena MHD test suite.
Details are available at:

https://www.astro.princeton.edu/~jstone/Athena/tests/field-loop/Field-loop.html

This case deals only with a line current in the +z direction (out of the
screen). +x is to the right, +y is up.

NOTE: This version of the code solves *only* the equations for Bx and By.

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

    0: Bx (x-component of magnetic field)
    1: By (y-component of magnetic field)

NOTE: These equations were last verified on 2023-02-05.

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
dependent_variable_names = ['Bx', 'By']

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = ["$B_x$", "$B_y$"]

# Number of dependent variables.
n_var = len(dependent_variable_names)

# Define the constant fluid flow field.
Q = 60.0
u0 = 1.0
ux = u0*np.sin(np.radians(Q))
uy = u0*np.cos(np.radians(Q))


# NOTE: In the functions defined below for the differential equations, the
# arguments can be unpacked as follows:
# def pde_XXX(X, Y, del_Y):
#     nX = X.shape[0]
#     t = tf.reshape(X[:, 0], (nX, 1))
#     x = tf.reshape(X[:, 1], (nX, 1))
#     y = tf.reshape(X[:, 2], (nX, 1))
#     (Bx, By) = Y
#     (del_Bx, del_By) = del_Y
#     dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
#     dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
#     dBx_dy = tf.reshape(del_Bx[:, 2], (nX, 1))
#     dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
#     dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
#     dBy_dy = tf.reshape(del_By[:, 2], (nX, 1))


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
    # (Bx, By) = Y
    (del_Bx, del_By) = del_Y
    dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
    dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
    dBx_dy = tf.reshape(del_Bx[:, 2], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
    # dBy_dy = tf.reshape(del_By[:, 2], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dBx_dt + ux*dBx_dx + uy*dBx_dy
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
    # (Bx, By) = Y
    (del_Bx, del_By) = del_Y
    # dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
    # dBx_dy = tf.reshape(del_Bx[:, 2], (nX, 1))
    dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
    dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
    dBy_dy = tf.reshape(del_By[:, 2], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dBy_dt + ux*dBy_dx + uy*dBy_dy
    return G


# Make a list of all of the differential equations.
de = [
    pde_Bx,
    pde_By,
]


if __name__ == "__main__":
    print("independent_variable_names = %s" % independent_variable_names)
    print("independent_variable_labels = %s" % independent_variable_labels)
    print("n_dim = %s" % n_dim)
    print("dependent_variable_names = %s" % dependent_variable_names)
    print("dependent_variable_labels = %s" % dependent_variable_labels)
    print("n_var = %s" % n_var)

    print("Q = %s" % Q)
    print("u0 = %s" % u0)
    print("ux = %s" % ux)
    print("uy = %s" % uy)
