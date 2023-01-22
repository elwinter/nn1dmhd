"""Problem definition file for a simple 1-D MHD problem.

This problem definition file describes an electron plasma wave, using
a set of partial differential equations for the density perturbation n1,
x-component of the velocity perturbation v1x, and x-component of the
electric field perturbation E1x. This version of the problem contains 2
frequency components, which should exhibit dispersion as the waves propagate.

NOTE: The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by the TensorFlow
code.

NOTE: In all code, below, the following indices are assigned to physical
variables (all are perturbations to steady-state values):

0: n1    # electron number density perturbation
1: u1x   # x-component of electron velocity perturbation
1: E1x   # x-component of electric field perturbation

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
from nn1dmhd import plasma


# Names of independent variables.
independent_variable_names = ["x", "t"]

# Number of independent variables.
ndim = len(independent_variable_names)

# Names of dependent variables.
dependent_variable_names = ["n1", "v1x", "E1x"]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# Define the problem domain.
x0 = 0.0
x1 = 1.0
t0 = 0.0
t1 = 1.0

# Normalized physical constants.
e = 1.0     # Unit charge
kb = 1.0    # Boltzmann constant
eps0 = 1.0  # Permeability of free space
me = 1.0    # Electron mass

# Adiabatic index for 1-D gas.
gamma = 3.0

# Ambient temperature (normalized to unit physical constants).
T = 1.0

# Wavelength and wavenumber of initial n/vx/Ex perturbations.
wavelengths = np.array([0.5, 1.0, 2.0])
kx = 2*np.pi/wavelengths
nc = len(kx)  # Number of wave components.

# Compute the electron thermal speed (independent of components).
vth = plasma.electron_thermal_speed(T, normalize=True)

# Steady-state value and perturbation amplitudes for number density.
n0 = 1.0
n1_amp = np.array([0.1, 0.1, 0.1])

# Compute the electron plasma angular frequency (independent of components).
wp = plasma.electron_plasma_angular_frequency(n0, normalize=True)

# Compute the electron plasma wave angular frequency for each component.                           
w = plasma.electron_plasma_wave_angular_frequency(n0, T, kx, normalize=True)

# Compute the wave phase speed for each component.
vphase = plasma.electron_plasma_wave_phase_speed(n0, T, kx, normalize=True)

# Compute the spin-up time to create the initial conditions. This is the time
# it takes for the slowest wave component to propagate to the other side.
TL = (x1 - x0)/np.min(vphase)

# Steady-state value and perturbation amplitudes for x-velocity.
v1x0 = 0.0
v1x_amp = w/kx*n1_amp/n0

# Steady-state value and perturbation amplitudes for x-electric field.
E1x0 = 0.0
E1x_amp = e*n1_amp/(kx*eps0)


def n1a(xt):
    """Compute the analytical solution for the density perturbation.

    Compute the anaytical solution for the density perturbation as a
    superposition of components of different frequencies.

    Parameters
    ----------
    xt : nd.array of float, shape (n, 2)
        x- and t-values for computation.

    Returns
    -------
    n1 : nd.array of float, shape (n,)
        Analytical values for density perturbation.
    """
    x = xt[:, 0]
    t = xt[:, 1]
    n1 = np.zeros_like(x)
    for (n1i, ki, wi) in zip(n1_amp, kx, w):
        n1 += n1i*np.sin(ki*x - wi*t)
    return n1


def v1xa(xt):
    """Compute the analytical solution for the x-velocity perturbation.

    Compute the anaytical solution for the x-velocity perturbation as a
    superposition of components of different frequencies.

    Parameters
    ----------
    xt : nd.array of float, shape (n, 2)
        x- and t-values for computation.

    Returns
    -------
    v1x : nd.array of float, shape (n,)
        Analytical values for x-velocity perturbation.
    """
    x = xt[:, 0]
    t = xt[:, 1]
    v1x = np.zeros_like(x)
    for (v1xi, ki, wi) in zip(v1x_amp, kx, w):
        v1x += v1xi*np.sin(ki*x - wi*t)
    return v1x


def E1xa(xt):
    """Compute the analytical solution for the x-electric field perturbation.

    Compute the anaytical solution for the x-electric field perturbation as a
    superposition of components of different frequencies.

    Parameters
    ----------
    xt : nd.array of float, shape (n, 2)
        x- and t-values for computation.

    Returns
    -------
    v1x : nd.array of float, shape (n,)
        Analytical values for x-velocity perturbation.
    """
    x = xt[:, 0]
    t = xt[:, 1]
    E1x = np.zeros_like(x)
    for (E1xi, ki, wi) in zip(E1x_amp, kx, w):
        E1x += E1xi*np.sin(ki*x - wi*t + np.pi/2)
    return E1x


def create_training_data_gridded(nx:int, nt:int):
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
Ya = [n1a, v1xa, E1xa]


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
    n = len(xt)
    bc = np.empty((n, n_var))
    for (i, ya) in enumerate(Ya):
        bc[:, i] = ya(xt)
    return bc


# Define the differential equations to solve using TensorFlow operations.

# @tf.function
def pde_n1(xt, Y1, del_Y1):
    """Differential equation for n1.

    Evaluate the differential equation for n1 (number density perturbation).

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y1 : list of nvar tf.Tensor, each shape (n, 1)
        Perturbations of dependent variables at each training point.
    del_Y1 : list of nvar tf.Tensor, each shape (n, 2)
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
    # (n1, v1x, E1x) = Y1
    (del_n1, del_v1x, del_E1x) = del_Y1
    # dn1_dx = tf.reshape(del_n1[:, 0], (n, 1))
    dn1_dt = tf.reshape(del_n1[:, 1], (n, 1))
    dv1x_dx = tf.reshape(del_v1x[:, 0], (n, 1))
    # dv1x_dt = tf.reshape(del_v1x[:, 1], (n, 1))
    # dE1x_dx = tf.reshape(del_E1x[:, 0], (n, 1))
    # dE1x_dt = tf.reshape(del_E1x[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = dn1_dt + n0*dv1x_dx
    return G


# @tf.function
def pde_v1x(xt, Y1, del_Y1):
    """Differential equation for v1x.

    Evaluate the differential equation for v1x (x-velocity perturbation).

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
    (n1, v1x, E1x) = Y1
    (del_n1, del_v1x, del_E1x) = del_Y1
    dn1_dx = tf.reshape(del_n1[:, 0], (n, 1))
    # dn1_dt = tf.reshape(del_n1[:, 1], (n, 1))
    # dv1x_dx = tf.reshape(del_v1x[:, 0], (n, 1))
    dv1x_dt = tf.reshape(del_v1x[:, 1], (n, 1))
    # dE1x_dx = tf.reshape(del_E1x[:, 0], (n, 1))
    # dE1x_dt = tf.reshape(del_E1x[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = dv1x_dt + e/me*E1x + gamma*kb*T/(me*n0)*dn1_dx
    return G


# @tf.function
def pde_E1x(xt, Y1, del_Y1):
    """Differential equation for E1x.

    Evaluate the differential equation for E1x (x-electric field perturbation).

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
    (n1, v1x, E1x) = Y1
    (del_n1, del_v1x, del_E1x) = del_Y1
    # dn1_dx = tf.reshape(del_n1[:, 0], (n, 1))
    # dn1_dt = tf.reshape(del_n1[:, 1], (n, 1))
    # dv1x_dx = tf.reshape(del_v1x[:, 0], (n, 1))
    # dv1x_dt = tf.reshape(del_v1x[:, 1], (n, 1))
    dE1x_dx = tf.reshape(del_E1x[:, 0], (n, 1))
    # dE1x_dt = tf.reshape(del_E1x[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = dE1x_dx + e/eps0*n1
    return G


# Make a list of all of the differential equations.
differential_equations = [
    pde_n1,
    pde_v1x,
    pde_E1x
]
