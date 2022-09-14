"""Problem definition file for a simple 1-D MHD problem.

This problem definition file describes an Alfven wave: unit pressure and
density, with a constant axial magnetic field (B0x = constant).

The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by the TensorFlow
code.

NOTE: The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by the TensorFlow
code.

NOTE: In all code, below, the following indices are assigned to physical
variables (all are perturbations to steady-state values):

0: rho1  # mass density perturbation
1: v1x   # x-component of velocity perturbation
2: v1y   # y-component of velocity perturbation
3: v1z   # z-component of velocity perturbation
4: B1y   # y-component of magnetic field perturbation
5: B1z   # z-xomponent of magnetic field perturbation

Pressure P is a function of the initial conditions and rho.

These equations are derived from the ideal MHD equations developed in Russel
et al, applying the assumptions used for Alfven waves (sections 3.5, 3.6 in
Russell et al, 2018).

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard modules.
import numpy as np

# Import supplemental modules.
import tensorflow as tf

# Import project modules.
import plasma


# Names of independent variables.
independent_variable_names = ["x", "t"]

# Number of independent variables.
ndim = len(independent_variable_names)

# Names of dependent variables.
dependent_variable_names = ["rho1", "v1x", "v1y", "v1z", "B1y", "B1z"]

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

# Ambient pressure (normalized to unit physical constants).
P0 = 1.0

# Wavelength and wavenumber of initial perturbations.
wavelengths = np.array([1.0])
kx = 2*np.pi/wavelengths
nc = len(kx)  # Number of wave components.

# Compute the electron thermal speed (independent of components).
vth = plasma.electron_thermal_speed(T, normalize=True)

# Steady-state value and perturbation amplitudes for mass density.
rho0 = 1.0
rho1_amp = 0.0

# Compute the electron plasma angular frequency (independent of components).
wp = plasma.electron_plasma_angular_frequency(rho0, normalize=True)

# Compute the electron plasma wave angular frequency for each component.                           
w = plasma.electron_plasma_wave_angular_frequency(rho0, T, kx, normalize=True)

# Compute the wave phase speed for each component.
vphase = plasma.electron_plasma_wave_phase_speed(rho0, T, kx, normalize=True)

# Steady-state value and perturbation amplitude for x-velocity.
v1x0 = 0.0
v1x_amp = 0.0

# Steady-state value and perturbation amplitude for y-velocity.
v1y0 = 0.0
v1y_amp = 0.1

# Steady-state value and perturbation amplitude for z-velocity.
v1z0 = 0.0
v1z_amp = 0.0

# Steady-state value and perturbation amplitude for x-magnetic field.
B0x = 1.0
B1x_amp = 0.0

# Steady-state value and perturbation amplitude for y-magnetic field.
B1y0 = 0.0
B1y_amp = 0.1

# Steady-state value and perturbation amplitude for z-magnetic field.
B1z0 = 0.0
B1z_amp = 0.0

# Alfven speed.
C_alfven = B0x/np.sqrt(rho0)

# Frequency and angular frequency of initial perturbation.
f = C_alfven/wavelengths
w = 2*np.pi*f


def create_training_data(nx, nt):
    """Create the training data.

    Create and return a set of training data of points evenly spaced in x and
    t. Flatten the data to a list of pairs of points. Also return copies
    of the data containing only internal points, and only boundary points.

    Parameters
    ----------
    nx, nt : int
        Number of points in x- and t- dimensions.

    Returns
    -------
    xt : np.ndarray, shape (nx*nt, 2)
        Array of all [x, t] points.
    xt_in : np.ndarray, shape ((nx - 1)*(nt - 2)), 2)
        Array of all [x, t] points within boundary.
    xt_bc : np.ndarray, shape (nx + 2*(nt - 1), 2)
        Array of all [x, t] points at boundary.
    """
    # Create the array of all training points (x, t), looping over t then x.
    x = np.linspace(x0, x1, nx)
    t = np.linspace(t0, t1, nt)
    X = np.repeat(x, nt)
    T = np.tile(t, nx)
    xt = np.vstack([X, T]).T

    # Now split the training data into two groups - inside the BC, and on the
    # BC.
    # Initialize the mask to keep everything.
    mask = np.ones(len(xt), dtype=bool)
    # Mask off the points at x = 0.
    mask[:nt] = False
    # Mask off the points at x = 1.
    mask[-nt:] = False
    # Mask off the points at t = 0.
    mask[::nt] = False

    # Extract the internal points.
    xt_in = xt[mask]

    # Invert the mask and extract the boundary points.
    mask = np.logical_not(mask)
    xt_bc = xt[mask]
    return xt, xt_in, xt_bc


def compute_boundary_conditions(xt):
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
    for (i, (x, t)) in enumerate(xt):
        if np.isclose(x, x0):
            # Periodic perturbation at x = 0.
            bc[i, :] = [
                rho0,
                v1x0,
                v1y_amp*np.sin(-w*t),
                v1z0,
                B1y_amp*np.sin(-w*t - np.pi),
                B1z0
            ]
        elif np.isclose(x, x1):
            # Periodic perturbation at x = 1, same as at x = 0.
            bc[i, :] = [
                rho0,
                v1x0,
                v1y_amp*np.sin(-w*t),
                v1z0,
                B1y_amp*np.sin(-w*t - np.pi),
                B1z0
            ]
        elif np.isclose(t, t0):
            bc[i, :] = [
                rho0,
                v1x0,
                v1y_amp*np.sin(kx*x),
                v1z0,
                B1y_amp*np.sin(kx*x + np.pi),
                B1z0
            ]
        else:
            raise ValueError
    return bc


# Define the differential equations using TensorFlow operations.

# @tf.function
def pde_rho1(xt, Y1, del_Y1):
    """Differential equation for rho1.

    Evaluate the differential equation for rho1 (density perturbation).

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
    # Each of these Tensors is shape (n, 1).
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    (rho1, v1x, v1y, v1z, B1y, B1z) = Y1
    (del_rho1, del_v1x, del_v1y, del_v1z, del_B1y, del_B1z) = del_Y1
    # drho1_dx = tf.reshape(del_rho1[:, 0], (n, 1))
    drho1_dt = tf.reshape(del_rho1[:, 1], (n, 1))
    dv1x_dx = tf.reshape(del_v1x[:, 0], (n, 1))
    # dv1x_dt = tf.reshape(del_v1x[:, 1], (n, 1))
    # dv1y_dx = tf.reshape(del_v1y[:, 0], (n, 1))
    # dv1y_dt = tf.reshape(del_v1y[:, 1], (n, 1))
    # dv1z_dx = tf.reshape(del_v1z[:, 0], (n, 1))
    # dv1z_dt = tf.reshape(del_v1z[:, 1], (n, 1))
    # dB1y_dx = tf.reshape(del_B1y[:, 0], (n, 1))
    # dB1y_dt = tf.reshape(del_B1y[:, 1], (n, 1))
    # dB1z_dx = tf.reshape(del_B1z[:, 0], (n, 1))
    # dB1z_dt = tf.reshape(del_B1z[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = drho1_dt + rho0*dv1x_dx
    return G


# @tf.function
def pde_v1x(xt, Y1, del_Y1):
    """Differential equation for v1x.

    Evaluate the differential equation for v1x (x-component of velocity
    perturbation).

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
    # Each of these Tensors is shape (n, 1).
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    # (rho1, v1x, v1y, v1z, B1y, B1z) = Y1
    (del_rho1, del_v1x, del_v1y, del_v1z, del_B1y, del_B1z) = del_Y1
    drho1_dx = tf.reshape(del_rho1[:, 0], (n, 1))
    # drho1_dt = tf.reshape(del_rho1[:, 1], (n, 1))
    # dv1x_dx = tf.reshape(del_v1x[:, 0], (n, 1))
    dv1x_dt = tf.reshape(del_v1x[:, 1], (n, 1))
    # dv1y_dx = tf.reshape(del_v1y[:, 0], (n, 1))
    # dv1y_dt = tf.reshape(del_v1y[:, 1], (n, 1))
    # dv1z_dx = tf.reshape(del_v1z[:, 0], (n, 1))
    # dv1z_dt = tf.reshape(del_v1z[:, 1], (n, 1))
    # dB1y_dx = tf.reshape(del_B1y[:, 0], (n, 1))
    # dB1y_dt = tf.reshape(del_B1y[:, 1], (n, 1))
    # dB1z_dx = tf.reshape(del_B1z[:, 0], (n, 1))
    # dB1z_dt = tf.reshape(del_B1z[:, 1], (n, 1))

    # Compute the presure derivative from the density derivative.
    dP1_dx = gamma*P0/rho0*drho1_dx

    # G is a Tensor of shape (n, 1).
    G = rho0*dv1x_dt + dP1_dx
    return G


# @tf.function
def pde_v1y(xt, Y1, del_Y1):
    """Differential equation for v1y.

    Evaluate the differential equation for v1y (y-component of velocity
    perturbation).

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
    # Each of these Tensors is shape (n, 1).
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    # (rho1, v1x, v1y, v1z, B1y, B1z) = Y1
    (del_rho1, del_v1x, del_v1y, del_v1z, del_B1y, del_B1z) = del_Y1
#    (del_v1y, del_B1y) = del_Y1
    # drho1_dx = tf.reshape(del_rho1[:, 0], (n, 1))
    # drho1_dt = tf.reshape(del_rho1[:, 1], (n, 1))
    # dv1x_dx = tf.reshape(del_v1x[:, 0], (n, 1))
    # dv1x_dt = tf.reshape(del_v1x[:, 1], (n, 1))
    # dv1y_dx = tf.reshape(del_v1y[:, 0], (n, 1))
    dv1y_dt = tf.reshape(del_v1y[:, 1], (n, 1))
    # dv1z_dx = tf.reshape(del_v1z[:, 0], (n, 1))
    # dv1z_dt = tf.reshape(del_v1z[:, 1], (n, 1))
    dB1y_dx = tf.reshape(del_B1y[:, 0], (n, 1))
    # dB1y_dt = tf.reshape(del_B1y[:, 1], (n, 1))
    # dB1z_dx = tf.reshape(del_B1z[:, 0], (n, 1))
    # dB1z_dt = tf.reshape(del_B1z[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = rho0*dv1y_dt - B0x*dB1y_dx
    return G


# @tf.function
def pde_v1z(xt, Y1, del_Y1):
    """Differential equation for v1z.

    Evaluate the differential equation for v1z (z-component of velocity
    perturbation).

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
    # Each of these Tensors is shape (n, 1).
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    # (rho1, v1x, v1y, v1z, B1y, B1z) = Y1
    (del_rho1, del_v1x, del_v1y, del_v1z, del_B1y, del_B1z) = del_Y1
    # drho1_dx = tf.reshape(del_rho1[:, 0], (n, 1))
    # drho1_dt = tf.reshape(del_rho1[:, 1], (n, 1))
    # dv1x_dx = tf.reshape(del_v1x[:, 0], (n, 1))
    # dv1x_dt = tf.reshape(del_v1x[:, 1], (n, 1))
    # dv1y_dx = tf.reshape(del_v1y[:, 0], (n, 1))
    # dv1y_dt = tf.reshape(del_v1y[:, 1], (n, 1))
    # dv1z_dx = tf.reshape(del_v1z[:, 0], (n, 1))
    dv1z_dt = tf.reshape(del_v1z[:, 1], (n, 1))
    # dB1y_dx = tf.reshape(del_B1y[:, 0], (n, 1))
    # dB1y_dt = tf.reshape(del_B1y[:, 1], (n, 1))
    dB1z_dx = tf.reshape(del_B1z[:, 0], (n, 1))
    # dB1z_dt = tf.reshape(del_B1z[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = rho0*dv1z_dt - B0x*dB1z_dx
    return G


# @tf.function
def pde_B1y(xt, Y1, del_Y1):
    """Differential equation for B1y.

    Evaluate the differential equation for B1y (y-component of magnetic
    field perturbation).

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
    # Each of these Tensors is shape (n, 1).
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    # (rho1, v1x, v1y, v1z, B1y, B1z) = Y1
    (del_rho1, del_v1x, del_v1y, del_v1z, del_B1y, del_B1z) = del_Y1
#    (del_v1y, del_B1y) = del_Y1
    # drho1_dx = tf.reshape(del_rho1[:, 0], (n, 1))
    # drho1_dt = tf.reshape(del_rho1[:, 1], (n, 1))
    # dv1x_dx = tf.reshape(del_v1x[:, 0], (n, 1))
    # dv1x_dt = tf.reshape(del_v1x[:, 1], (n, 1))
    dv1y_dx = tf.reshape(del_v1y[:, 0], (n, 1))
    # dv1y_dt = tf.reshape(del_v1y[:, 1], (n, 1))
    # dv1z_dx = tf.reshape(del_v1z[:, 0], (n, 1))
    # dv1z_dt = tf.reshape(del_v1z[:, 1], (n, 1))
    # dB1y_dx = tf.reshape(del_B1y[:, 0], (n, 1))
    dB1y_dt = tf.reshape(del_B1y[:, 1], (n, 1))
    # dB1z_dx = tf.reshape(del_B1z[:, 0], (n, 1))
    # dB1z_dt = tf.reshape(del_B1z[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = dB1y_dt - B0x*dv1y_dx
    return G


# @tf.function
def pde_B1z(xt, Y1, del_Y1):
    """Differential equation for B1z.

    Evaluate the differential equation for B1z (z-component of magnetic
    field perturbation).

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
    # Each of these Tensors is shape (n, 1).
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    # (rho1, v1x, v1y, v1z, B1y, B1z) = Y1
    (del_rho1, del_v1x, del_v1y, del_v1z, del_B1y, del_B1z) = del_Y1
    # drho1_dx = tf.reshape(del_rho1[:, 0], (n, 1))
    # drho1_dt = tf.reshape(del_rho1[:, 1], (n, 1))
    # dv1x_dx = tf.reshape(del_v1x[:, 0], (n, 1))
    # dv1x_dt = tf.reshape(del_v1x[:, 1], (n, 1))
    # dv1y_dx = tf.reshape(del_v1y[:, 0], (n, 1))
    # dv1y_dt = tf.reshape(del_v1y[:, 1], (n, 1))
    dv1z_dx = tf.reshape(del_v1z[:, 0], (n, 1))
    # dv1z_dt = tf.reshape(del_v1z[:, 1], (n, 1))
    # dB1y_dx = tf.reshape(del_B1y[:, 0], (n, 1))
    # dB1y_dt = tf.reshape(del_B1y[:, 1], (n, 1))
    # dB1z_dx = tf.reshape(del_B1z[:, 0], (n, 1))
    dB1z_dt = tf.reshape(del_B1z[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = dB1z_dt - B0x*dv1z_dx
    return G


# Make a list of all of the differential equations.
differential_equations = [
    pde_rho1,
    pde_v1x,
    pde_v1y,
    pde_v1z,
    pde_B1y,
    pde_B1z,
]
