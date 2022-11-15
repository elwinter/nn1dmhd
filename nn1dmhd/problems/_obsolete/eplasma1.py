"""Problem definition file for a simple 1-D MHD problem.

This problem definition file describes an electron plasma wave, using
a single equation for the density perturbation n1. This is possible
because in the ideal 1-D MHD case, the x-velocity perturbation v1x,
and the x-component of the electric field (E1x) are both functions of
the density perturbation n1.

NOTE: The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by the TensorFlow
code.

NOTE: In all code, below, the following indices are assigned to physical
variables (all are perturbations to initial values):

0: n1    # electron number density perturbation

These equations are derived from the ideal MHD equations developed in
Russel et al, applying the assumptions used for electron plasma waves
(see Greenwald notes).

Author
------
Eric Winter (eric.winter62@gmail.com)

"""


# Import standard modules.
import numpy as np

# Import supplemental modules.
import tensorflow as tf

# Import project modules.
import nn1dmhd.plasma as plasma


# Names of independent variables.
independent_variable_names = ["x", "t"]

# Number of independent variables.
ndim = len(independent_variable_names)

# Names of dependent variables.
variable_names = ["n1"]

# Number of dependent variables.
n_var = len(variable_names)


# Define the problem domain.
x0 = 0.0
x1 = 1.0
t0 = 0.0
t1 = 1.0

# Ambient temperature (normalized to unit physical constants).
T = 1.0

# Wavelength and wavenumber of initial density/velocity/Ex perturbation.
wavelength = 1.0
kx = 2*np.pi/wavelength

# Ambient density and density perturbation amplitude at t = 0, for all x.
n0 = 1.0
n10 = 0.1


# Compute the electron plasma angular frequency (independent of components).
wp = plasma.electron_plasma_angular_frequency(n0, normalize=True)

# Compute the electron thermal speed (independent of components).
vth = plasma.electron_thermal_speed(T, normalize=True)

# Compute the electron plasma wave angular frequency for each component.                           
w = plasma.electron_plasma_wave_angular_frequency(n0, T, kx, normalize=True)

# Compute the wave phase speed for each component.
vphase = plasma.electron_plasma_wave_phase_speed(n0, T, kx, normalize=True)


def n1a(xt):
    """Compute the analytical solution.

    Compute the anaytical solution.

    Parameters
    ----------
    xt : nd.array of float, shape (n, 2)
        x- and t-values for computation.

    Returns
    -------
    n1 : nd.array of float, shape (n,)
    """
    x = xt[:, 0]
    t = xt[:, 1]
    n1 = n10*np.sin(kx*x - w*t)
    return n1


def create_training_data(nx:int, nt:int):
    """Create the training data.

    Create and return a set of training data of points evenly spaced in x and
    t. Flatten the data to a list of pairs of points. Also return copies
    of the data containing only internal points, and only boundary points.

    The boundary conditions are defined at (x, t) = (x, 0) and (x, t) = (0, t).

    Parameters
    ----------
    nx, nt : int
        Number of points in x- and t- dimensions.

    Returns
    -------
    xt : np.ndarray, shape (nx*nt, 2)
        Array of all [x, t] points.
    xt_in : np.ndarray, shape (nx*nt - n_bc, 2)
        Array of all [x, t] points within boundary.
    xt_bc : np.ndarray, shape (nx + nt - 1, 2)
        Array of all [x, t] points at boundary.
    """
    # Create the array of all training points (x, t), looping over t then x.
    x = np.linspace(x0, x1, nx)
    t = np.linspace(t0, t1, nt)
    X = np.repeat(x, nt)
    T = np.tile(t, nx)
    xt = np.vstack([X, T]).T

    # Now split the training data into two groups - inside the boundary,
    # and on the boundary.

    # Initialize the mask to keep everything.
    mask = np.ones(len(xt), dtype=bool)

    # Mask off the points at x = 0.
    mask[:nt] = False

    # Mask off the points at t = 0.
    mask[::nt] = False

    # Extract the internal points.
    xt_in = xt[mask]

    # Invert the mask and extract the boundary points.
    mask = np.logical_not(mask)
    xt_bc = xt[mask]

    # Return the lists of training points.
    return xt, xt_in, xt_bc


# List the analytical solutions so they can also be used for computing the
# boundary conditions.
Ya = [n1a]


def compute_boundary_conditions(xt:np.ndarray):
    """Compute the boundary conditions.

    Parameters
    ----------
    xt : np.ndarray of float
        Values of (x, t) on the boundaries, shape (n_bc, 2)

    Returns
    -------
    bc : np.ndarray of float, shape (n_bc, n_var)
        Values of each dependent variable on boundary.
    """
    bc = Ya[0](xt)
    # Since only a single variable, added a dummy column dimension.
    bc = bc[:, np.newaxis]
    return bc


# Define the differential equations using TensorFlow operations.

# @tf.function
def pde_n1(xt, Y1, del_Y1):
    """Differential equation for n1.

    Evaluate the differential equation for n1 (number density perturbation).

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y1 : list of n_var tf.Tensor, each shape (n, 1)
        Perturbations of dependent variables at each training point.
    del_Y1 : list of n_var tf.Tensor, each shape (n, 2)
        Values of gradients of Y1 wrt (x, t) at each training point.

    Returns
    -------
    G : tf.Tensor, shape(n, 1)
        Value of differential equation at each training point.
    """
    n = xt.shape[0]
    # Each of these Tensors is shape (n, 1).
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    # (n1,) = Y1
    # (del_n1,) = del_Y1
    (del_n1,) = del_Y1
    dn1_dx = tf.reshape(del_n1[:, 0], (n, 1))
    dn1_dt = tf.reshape(del_n1[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = dn1_dt + w/kx*dn1_dx
    return G


# Make a list of all of the differential equations.
differential_equations = [
    pde_n1,
]
