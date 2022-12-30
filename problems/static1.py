"""Problem definition file for a simple 1-D MHD problem.

This problem definition file describes a static situation: unit pressure and
density, all else is 0. The run should just converge to the initial values for
each quantity.

NOTE: These equations were verified on 2022-11-18.

NOTE: The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by the TensorFlow
code.

NOTE: In all code, below, the following indices are assigned to physical
independent variables:

    0: t
    1: x

NOTE: In all code, below, the following indices are assigned to physical
dependent variables:

    0: n
    1: P
    2: ux
    3: uy
    4: uz
    5: Bx
    6: By
    7: Bz

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
independent_variable_names = ['t', 'x']

# Labels for independent variables (may use LaTex) - use for plots.
independent_variable_labels = ['$t$', '$x$']

# Number of problem dimensions (independent variables).
n_dim = len(independent_variable_names)

# Names of dependent variables.
dependent_variable_names = ['n', 'P', 'ux', 'uy', 'uz', 'Bx', 'By', 'Bz']

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = [
    '$n$', '$P$',
    '$u_x$', '$u_y$', '$u_z$',
    '$B_x$', '$B_y$', '$B_z$'
]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# Define the problem domain.
t0 = 0.0
t1 = 1.0
x0 = 0.0
x1 = 1.0
domain = np.array(
    [[t0, t1],
     [x0, x1]]
)

# Adiabatic index = (N + 2)/N, N = # DOF = 1.
ɣ = 3/2

# Normalized physical constants.
μ0 = 1.0  # Permeability of free space
me = 1.0  # Electron mass

# Initial values for each dependent variable (dimensionless).
n0 = 1.0
P0 = 1.0
ux0 = 0.0
uy0 = 0.0
uz0 = 0.0
Bx0 = 0.0
By0 = 0.0
Bz0 = 0.0
initial_values = [n0, P0, ux0, uy0, uz0, Bx0, By0, Bz0]


def n_analytical(X: np.ndarray):
    """Compute analytical solution for number density.

    Compute anaytical solution for number density.

    Parameters
    ----------
    X : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    n : np.ndarray of float, shape (n,)
        Analytical values for number density.
    """
    n = np.full((X.shape[0],), n0)
    return n


def P_analytical(X: np.ndarray):
    """Compute analytical solution for pressure.

    Compute anaytical solution for pressure.

    Parameters
    ----------
    X : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    P : np.ndarray of float, shape (n,)
        Analytical values for pressure.
    """
    P = np.full((X.shape[0],), P0)
    return P


def ux_analytical(X: np.ndarray):
    """Compute analytical solution for x-velocity.

    Compute anaytical solution for x-velocity.

    Parameters
    ----------
    X : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    ux : np.ndarray of float, shape (n,)
        Analytical values for x-velocity.
    """
    ux = np.full((X.shape[0],), ux0)
    return ux


def uy_analytical(X: np.ndarray):
    """Compute analytical solution for y-velocity.

    Compute anaytical solution for y-velocity.

    Parameters
    ----------
    X : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    uy : np.ndarray of float, shape (n,)
        Analytical values for y-velocity.
    """
    uy = np.full((X.shape[0],), uy0)
    return uy


def uz_analytical(X: np.ndarray):
    """Compute analytical solution for z-velocity.

    Compute anaytical solution for z-velocity.

    Parameters
    ----------
    X : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    uz : np.ndarray of float, shape (n,)
        Analytical values for z-velocity.
    """
    uz = np.full((X.shape[0],), uz0)
    return uz


def Bx_analytical(X: np.ndarray):
    """Compute analytical solution for x-magnetic field.

    Compute anaytical solution for x-magnetic field.

    Parameters
    ----------
    X : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    Bx : np.ndarray of float, shape (n,)
        Analytical values for x-magnetic field.
    """
    Bx = np.full((X.shape[0],), Bx0)
    return Bx


def By_analytical(X: np.ndarray):
    """Compute analytical solution for y-magnetic field.

    Compute anaytical solution for y-magnetic field.

    Parameters
    ----------
    X : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    By : np.ndarray of float, shape (n,)
        Analytical values for y-magnetic field.
    """
    By = np.full((X.shape[0],), By0)
    return By


def Bz_analytical(X: np.ndarray):
    """Compute analytical solution for z-magnetic field.

    Compute anaytical solution for z-magnetic field.

    Parameters
    ----------
    X : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    Bz : np.ndarray of float, shape (n,)
        Analytical values for z-magnetic field.
    """
    Bz = np.full((X.shape[0],), Bz0)
    return Bz


# Gather all of the analytical solutions into a list.
Y_analytical = [
    n_analytical,
    P_analytical,
    ux_analytical,
    uy_analytical,
    uz_analytical,
    Bx_analytical,
    By_analytical,
    Bz_analytical,
]


# @tf.function
def pde_n(X, Y, dY_dX):
    """Differential equation for number density.

    Evaluate the differential equation for number density. This equation is
    derived from the equation of mass continuity.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    dY_dX : list of n_var tf.Tensor, each shape (n, n_dim)
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
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = dY_dX
    dn_dt = tf.reshape(del_n[:, 0], (nX, 1))
    dn_dx = tf.reshape(del_n[:, 1], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, 0], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, 1], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, 0], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, 1], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, 0], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, 1], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, 0], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, 1], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 0], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, 1], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dn_dt + n*dux_dx + dn_dx*ux
    return G


# @tf.function
def pde_P(X, Y, dY_dX):
    """Differential equation for P.

    Evaluate the differential equation for pressure. This equation is derived
    from the equation of conservation of energy.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    dY_dX : list of n_var tf.Tensor, each shape (n, n_dim)
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
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = dY_dX
    dn_dt = tf.reshape(del_n[:, 0], (nX, 1))
    dn_dx = tf.reshape(del_n[:, 1], (nX, 1))
    dP_dt = tf.reshape(del_P[:, 0], (nX, 1))
    dP_dx = tf.reshape(del_P[:, 1], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, 0], (nX, 1))
    # dux_dx = tf.reshape(del_ux[:, 1], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, 0], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, 1], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, 0], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, 1], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 0], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, 1], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = -ɣ*P/n*(dn_dt + ux*dn_dx) + (dP_dt + ux*dP_dx)/me
    return G


# @tf.function
def pde_ux(X, Y, dY_dX):
    """Differential equation for x-velocity.

    Evaluate the differential equation for x-velocity. This equation is derived
    from the equation of conservation of x-momentum.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    dY_dX : list of n_var tf.Tensor, each shape (n, n_dim)
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
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = dY_dX
    # dn_dt = tf.reshape(del_n[:, 0], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, 1], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, 0], (nX, 1))
    dP_dx = tf.reshape(del_P[:, 1], (nX, 1))
    dux_dt = tf.reshape(del_ux[:, 0], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, 1], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, 0], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, 1], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, 0], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, 1], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
    dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 0], (nX, 1))
    dBz_dx = tf.reshape(del_Bz[:, 1], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = n*(dux_dt + ux*dux_dx) + dP_dx/me + (By*dBy_dx + Bz*dBz_dx)/(me*μ0)
    return G


# @tf.function
def pde_uy(X, Y, dY_dX):
    """Differential equation for y-velocity.

    Evaluate the differential equation for y-velocity. This equation is derived
    from the equation of conservation of y-momentum.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    dY_dX : list of n_var tf.Tensor, each shape (n, n_dim)
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
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = dY_dX
    # dn_dt = tf.reshape(del_n[:, 0], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, 1], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, 0], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, 1], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, 0], (nX, 1))
    # dux_dx = tf.reshape(del_ux[:, 1], (nX, 1))
    duy_dt = tf.reshape(del_uy[:, 0], (nX, 1))
    duy_dx = tf.reshape(del_uy[:, 1], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, 0], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, 1], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
    dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 0], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, 1], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = n*(duy_dt + ux*duy_dx) - Bx*dBy_dx/(me*μ0)
    return G


# @tf.function
def pde_uz(X, Y, dY_dX):
    """Differential equation for z-velocity.

    Evaluate the differential equation for z-velocity. This equation is derived
    from the equation of conservation of z-momentum.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    dY_dX : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables at
        each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.
    """
    nX = X.shape[0]
    t = tf.reshape(X[:, 0], (nX, 1))
    x = tf.reshape(X[:, 1], (nX, 1))
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = dY_dX
    # dn_dt = tf.reshape(del_n[:, 0], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, 1], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, 0], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, 1], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, 0], (nX, 1))
    # dux_dx = tf.reshape(del_ux[:, 1], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, 0], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, 1], (nX, 1))
    duz_dt = tf.reshape(del_uz[:, 0], (nX, 1))
    duz_dx = tf.reshape(del_uz[:, 1], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 0], (nX, 1))
    dBz_dx = tf.reshape(del_Bz[:, 1], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = n*(duz_dt + ux*duz_dx) - Bx*dBz_dx/(me*μ0)
    return G


# @tf.function
def pde_Bx(X, Y, dY_dX):
    """Differential equation for x-magnetic field.

    Evaluate the differential equation for x-magnetic field. This equation is
    derived from the x-component of Faraday's Law.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    dY_dX : list of n_var tf.Tensor, each shape (n, n_dim)
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
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = dY_dX
    # dn_dt = tf.reshape(del_n[:, 0], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, 1], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, 0], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, 1], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, 0], (nX, 1))
    # dux_dx = tf.reshape(del_ux[:, 1], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, 0], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, 1], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, 0], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, 1], (nX, 1))
    dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
    dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 0], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, 1], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dBx_dt + ux*dBx_dx
    return G


# @tf.function
def pde_By(X, Y, dY_dX):
    """Differential equation for y-magnetic field.

    Evaluate the differential equation for y-magnetic field. This equation is
    derived from the y-component of Faraday's Law.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    dY_dX : list of n_var tf.Tensor, each shape (n, n_dim)
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
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = dY_dX
    # dn_dt = tf.reshape(del_n[:, 0], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, 1], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, 0], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, 1], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, 0], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, 1], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, 0], (nX, 1))
    duy_dx = tf.reshape(del_uy[:, 1], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, 0], (nX, 1))
    # duz_dx = tf.reshape(del_uz[:, 1], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
    dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
    dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 0], (nX, 1))
    # dBz_dx = tf.reshape(del_Bz[:, 1], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dBy_dt + ux*dBy_dx + By*dux_dx - Bx*duy_dx
    return G


# @tf.function
def pde_Bz(X, Y, dY_dX):
    """Differential equation for z-magnetic field.

    Evaluate the differential equation for z-magnetic field. This equation is
    derived from the z-component of Faraday's Law.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    dY_dX : list of n_var tf.Tensor, each shape (n, n_dim)
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
    (n, P, ux, uy, uz, Bx, By, Bz) = Y
    (del_n, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = dY_dX
    # dn_dt = tf.reshape(del_n[:, 0], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, 1], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, 0], (nX, 1))
    # dP_dx = tf.reshape(del_P[:, 1], (nX, 1))
    # dux_dt = tf.reshape(del_ux[:, 0], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, 1], (nX, 1))
    # duy_dt = tf.reshape(del_uy[:, 0], (nX, 1))
    # duy_dx = tf.reshape(del_uy[:, 1], (nX, 1))
    # duz_dt = tf.reshape(del_uz[:, 0], (nX, 1))
    duz_dx = tf.reshape(del_uz[:, 1], (nX, 1))
    # dBx_dt = tf.reshape(del_Bx[:, 0], (nX, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 1], (nX, 1))
    # dBy_dt = tf.reshape(del_By[:, 0], (nX, 1))
    # dBy_dx = tf.reshape(del_By[:, 1], (nX, 1))
    dBz_dt = tf.reshape(del_Bz[:, 0], (nX, 1))
    dBz_dx = tf.reshape(del_Bz[:, 1], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dBz_dt + ux*dBz_dx + Bz*dux_dx - Bx*duz_dx
    return G


# Make a list of all of the differential equations.
de = [
    pde_n,
    pde_P,
    pde_ux,
    pde_uy,
    pde_uz,
    pde_Bx,
    pde_By,
    pde_Bz
]


if __name__ == "__main__":
    print("independent_variable_names = %s" % independent_variable_names)
    print("independent_variable_labels = %s" % independent_variable_labels)
    print("n_dim = %s" % n_dim)
    print("dependent_variable_names = %s" % dependent_variable_names)
    print("dependent_variable_labels = %s" % dependent_variable_labels)
    print("n_var = %s" % n_var)

    print("%s <= t <= %s" % (t0, t1))
    print("%s <= x <= %s" % (x0, x1))
    print("domain = %s" % domain)

    print("n0 = %s" % n0)
    print("P0 = %s" % P0)
    print("ux0 = %s" % ux0)
    print("uy0 = %s" % uy0)
    print("uz0 = %s" % uz0)
    print("Bx0 = %s" % Bx0)
    print("By0 = %s" % By0)
    print("Bz0 = %s" % Bz0)

    nt = 3
    nx = 4
    Xg, Xg_in, Xg_bc = create_training_data_gridded(nt, nx)
    print("Xg = %s" % Xg)
    print("Xg_in = %s" % Xg_in)
    print("Xg_bc = %s" % Xg_bc)

    bc = compute_boundary_conditions(Xg_bc)
    print("bc = %s" % bc)
