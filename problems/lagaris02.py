"""Problem definition file for simple ODE (Lagaris problem 2).

This ordinary differential equation is taken from the paper:

"Artificial Neural Networks for Solving Ordinary and Partial Differential
Equations", by Isaac Elias Lagaris, Aristidis Likas, and Dimitrios I. Fotiadis
IEEE Transactions on Neural Networks, Vol. 9, No. 5, September 1998 987

NOTE: The functions in this module are defined using TensorFlow operations, so
they can be used efficiently by the TensorFlow code.

NOTE: In all code, the following indices are used for independent variables:

    0: x

NOTE: In all code, the following indices are used for dependent variables:

    0: y

NOTE: For all methods:

X represents an a set of arbitrary evaluation points. It is a tf.Tensor with
shape (n, n_dim), where n is the number of points, and n_dim is the number of
dimensions (independent variables) in the problem. For an ODE, n_dim is 1,
giving a shape of (n, 1).

Y represents a set of dependent variables at each point in X. This variable is
a list of n_var tf.Tensor, each shape (n, 1), where n_var is the number of
dependent variables. For an ODE, n_var is 1, giving a list of 1 Tensor of
shape of (n, 1).

dY_dX contains the first derivatives of each dependent variable with respect
to each independent variable, at each point in X. It is a list of n_var
tf.Tensor, each shape (n, n_dim). For an ODE, n_var and n_dim are 1, for a
list of 1 Tensor of shape (n, 1).

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard modules.

# Import supplemental modules.
import tensorflow as tf

# Import project modules.


# Names of independent variables.
independent_variable_names = ['x']

# Labels for independent variables (may use LaTex) - use for plots.
independent_variable_labels = ["$x$"]

# Number of problem dimensions (independent variables).
n_dim = len(independent_variable_names)

# Names of dependent variables.
dependent_variable_names = ['y',]

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = ["$y$"]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# Define the problem domain.
x0 = 0.0
x1 = 1.0
domain = tf.reshape([x0, x1], (n_dim, 2))


def y_analytical(X):
    """Analytical solution to lagaris01.

    Analytical solution to lagaris01.

    Parameters
    ----------
    X : tf.Tensor, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    y : tf.Tensor, shape (n, 1)
        Analytical solution at each x-value.
    """
    n = X.shape[0]
    # x and y are shape (n,)
    x = X[:, 0]
    y = tf.math.exp(-x/5)*tf.math.sin(x)
    # Reshape to (n, 1)
    y = tf.reshape(y, (n, 1))
    return y


# Gather the analytical solutions in a list.
Y_analytical = [y_analytical]


def dy_dx_analytical(X):
    """Analytical 1st derivative to lagaris01.

    Analytical 1st derivative of y wrt x for lagaris01.

    Parameters
    ----------
    X : tf.Tensor, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    dy_dx : tf.Tensor, shape (n, 1)
        Analytical derivatives of y wrt x at each x-value.
    """
    n = X.shape[0]
    # x and dy_dx are shape (n,)
    x = X[:, 0]
    dy_dx = 0.2*tf.math.exp(-x/5)*(5.0*tf.math.cos(x) - tf.math.sin(x))
    # Reshape to (n, 1)
    dy_dx = tf.reshape(dy_dx, (n, 1))
    return dy_dx


# Gather the analytical derivatives in a list of lists.
dY_dX_analytical = [dy_dx_analytical]


def lagaris02(X, Y, dY_dX):
    """1st-order ODE (Example 2 from Lagaris et al (1998)).

    Compute the 1st-order ODE (Example 2 from Lagaris et al (1998)) in
    standard form.

    Parameters
    ----------
    X : tf.Tensor, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    dY_dX : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of derivatives of dependent variables wrt independent variables
        at each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.
    """
    n = X.shape[0]
    # x, y, dy_dx are shape (n,).
    x = X[:, 0]
    y = Y[0][:, 0]
    dy_dx = dY_dX[0][:, 0]
    G = G = dy_dx + y/5 - tf.math.exp(-x/5)*tf.math.cos(x)
    # G is shape (n,) so reshape to (n, 1).
    G = tf.reshape(G, (n, 1))
    return G


# Gather the differential equations into a list.
de = [lagaris02]


if __name__ == "__main__":
    print("independent_variable_names = %s" % independent_variable_names)
    print("independent_variable_labels = %s" % independent_variable_labels)
    print("n_dim = %s" % n_dim)
    print("dependent_variable_names = %s" % dependent_variable_names)
    print("dependent_variable_labels = %s" % dependent_variable_labels)
    print("n_var = %s" % n_var)

    print("%s <= x <= %s" % (x0, x1))
    print("domain = %s" % domain)

    n_x = 5
    # Must reshape (nx,) to (nx, n_dim) to get 2D shape.
    X = tf.reshape(tf.linspace(x0, x1, n_x), (n_x, n_dim))
    print("X = %s" % X)
    Y = [f(X) for f in Y_analytical]
    print("Y = %s" % Y)
    dY_dX = [f(X) for f in dY_dX_analytical]
    print("dY_dX = %s" % dY_dX)
    # Since X, Y, dY_dX are all analytical, all G should be ~0.
    G = lagaris02(X, Y, dY_dX)
    print("G = %s" % G)
