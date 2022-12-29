"""Problem definition file for a simple 1-D MHD problem.

This problem definition file describes an Alfven wave: unit pressure and
density, with a constant axial magnetic field (B0x = constant).

This problem only specifies the initial condition, so the result is a single
pulse propagating in t and x.

This problem uses the linearized MHD equations.

NOTE: The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by the TensorFlow
code.

NOTE: In all code, below, the following indices are assigned to physical
independent variables:

    0: t
    1: x

NOTE: In all code, below, the following indices are assigned to physical
dependent variables:

    0: ρ1
    1: P1
    2: u1x
    3: u1y
    4: u1z
    5: B1x
    6: B1y
    7: B1z

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard modules.

# Import supplemental modules.
import numpy as np
import tensorflow as tf

# Import project modules.
from nn1dmhd import plasma
from nn1dmhd.training_data import create_training_points_gridded


# Normalized physical constants.
𝑒 = 1.0     # Unit charge
𝑘b = 1.0    # Boltzmann constant
ε0 = 1.0  # Permeability of free space
𝑚e = 1.0    # Electron mass
μ0 = 1.0  # Permeability of free space

# Names of independent variables.
independent_variable_names = ['t', 'x']

# Labels for independent variables (may use LaTex) - use for plots.
independent_variable_labels = [r'$t$', r'$x$']

# Number of problem dimensions (independent variables).
n_dim = len(independent_variable_names)

# Names of dependent variables.
dependent_variable_names = [
    'ρ1', 'P1',
    'u1x', 'u1y', 'u1z',
    'B1x', 'B1y', 'B1z'
]

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = [
    r'$\rho_1$', r'$P_1$',
    r'$u_{1x}$', r'$u_{1y}$', r'$u_{1z}$',
    r'$B_{1x}$', r'$B_{1y}$', r'$B_{1z}$'
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


# Ambient plasma parameters

# Ambient temperature (normalized to unit physical constants).
T = 1.0

# Adiabatic index = (N + 2)/N, N = # DOF.
ɣ = 3.0

# Initial values for each dependent variable (dimensionless).
ρ0 = 1.0
P0 = 1.0
u0x = 0.0
u0y = 0.0
u0z = 0.0
B0x = 1.0
B0y = 0.0
B0z = 0.0
initial_values = [ρ0, P0, u0x, u0y, u0z, B0x, B0y, B0z]


# Plasma computed values

# Compute the electron thermal speed (independent of components).
vth = plasma.electron_thermal_speed(T, normalize=True)

# Compute the electron plasma angular frequency (independent of components).
ωp = plasma.electron_plasma_angular_frequency(ρ0, normalize=True)

# Alfven speed.
C_Alfven = B0x/np.sqrt(ρ0)


# Perturbations

# Perturbation amplitudes for each dependent variable (dimensionless).
ρ1_amp = 0.0
P1_amp = 0.0
u1x_amp = 0.0
u1y_amp = 0.1
u1z_amp = 0.0
B1x_amp = 0.0
B1y_amp = 0.1
B1z_amp = 0.0

# Wavelength and wavenumber of initial perturbations.
λ = 1.0
kx = 2*np.pi/λ

# Compute the electron plasma wave angular frequency for each component.
# ω = plasma.electron_plasma_wave_angular_frequency(ρ0, T, kx, normalize=True)

# Compute the wave phase speed for each component.
vphase = plasma.electron_plasma_wave_phase_speed(ρ0, T, kx, normalize=True)

# Frequency and angular frequency of initial perturbation.
f = C_Alfven/λ
ω = 2*np.pi*f


def ρ1_analytical(X: np.ndarray):
    """Compute analytical solution for mass density perturbation.

    Compute anaytical solution for mass density perturbation.

    Parameters
    ----------
    X : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    ρ1 : np.ndarray of float, shape (n,)
        Analytical values for mass density perturbation.
    """
    ρ1 = np.full((X.shape[0],), ρ1_amp)
    return ρ1


def P1_analytical(X: np.ndarray):
    """Compute analytical solution for pressure perturbation.

    Compute anaytical solution for pressure perturbation.

    Parameters
    ----------
    X : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    P1 : np.ndarray of float, shape (n,)
        Analytical values for pressure perturbation.
    """
    P1 = np.full((X.shape[0],), P1_amp)
    return P1


def u1x_analytical(X: np.ndarray):
    """Compute analytical solution for x-velocity perturbation.

    Compute anaytical solution for x-velocity perturbation.

    Parameters
    ----------
    X : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    u1x : np.ndarray of float, shape (n,)
        Analytical values for x-velocity perturbation.
    """
    u1x = np.full((X.shape[0],), u1x_amp)
    return u1x


def u1y_analytical(X: np.ndarray):
    """Compute analytical solution for y-velocity perturbation.

    Compute anaytical solution for y-velocity perturbation.

    Parameters
    ----------
    X : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    uy : np.ndarray of float, shape (n,)
        Analytical values for y-velocity perturbation.
    """
    t = X[:, 0]
    x = X[:, 1]
    u1y = u1y_amp*np.sin(kx*x - ω*t)
    return u1y


def u1z_analytical(X: np.ndarray):
    """Compute analytical solution for z-velocity perturbation.

    Compute anaytical solution for z-velocity perturbation.

    Parameters
    ----------
    X : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    u1z : np.ndarray of float, shape (n,)
        Analytical values for z-velocity perturbation.
    """
    u1z = np.full((X.shape[0],), u1z_amp)
    return u1z


def B1x_analytical(X: np.ndarray):
    """Compute analytical solution for x-magnetic field perturbation.

    Compute anaytical solution for x-magnetic field perturbation.

    Parameters
    ----------
    X : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    B1x : np.ndarray of float, shape (n,)
        Analytical values for x-magnetic field perturbation.
    """
    t = X[:, 0]
    x = X[:, 1]
    B1x = np.zeros((len(t),))
    return B1x


def B1y_analytical(X: np.ndarray):
    """Compute analytical solution for y-magnetic field perturbation.

    Compute anaytical solution for y-magnetic field perturbation.

    Parameters
    ----------
    X : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    B1y : np.ndarray of float, shape (n,)
        Analytical values for y-magnetic field perturbation.
    """
    t = X[:, 0]
    x = X[:, 1]
    B1y = B1y_amp*np.sin(kx*x - ω*t + np.pi)
    return B1y


def B1z_analytical(X: np.ndarray):
    """Compute analytical solution for z-magnetic field perturbation.

    Compute anaytical solution for z-magnetic field perturbation.

    Parameters
    ----------
    X : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    B1z : np.ndarray of float, shape (n,)
        Analytical values for z-magnetic field perturbation.
    """
    B1z = np.full((X.shape[0],), B1z_amp)
    return B1z


# Gather all of the analytical solutions into a list.
ψ_analytical = [
    ρ1_analytical,
    P1_analytical,
    u1x_analytical,
    u1y_analytical,
    u1z_analytical,
    B1x_analytical,
    B1y_analytical,
    B1z_analytical,
]


def create_training_data_gridded(nt: int, nx: int):
    """Create the training data on an evenly-spaced grid.

    Create and return a set of training points evenly spaced in x, and
    t. Flatten the data to a list of points. Also return copies of the data
    containing only internal points, and only boundary points.

    Boundary points occur where:

    t = t0

    Parameters
    ----------
    nt, nx : int
        Number of points in t-, and x-dimensions.

    Returns
    -------
    Xg : np.ndarray, shape (nt*nx, n_dim)
        Array of all training points.
    Xg_in : np.ndarray, shape (n_in, n_dim)
        Array of all training points within the boundary.
    Xg_bc : np.ndarray, shape (n_bc, n_dim)
        Array of all training points on the boundary.
    """
    # Create the training grid and mask.
    ng = [nt, nx]
    ng_total = np.prod(ng)
    Xg = create_training_points_gridded(ng, domain)
    mask = np.ones(ng_total, dtype=bool)

    # Compute the coordinates of each training point, and in-domain mask value.
    for (i, (t, x)) in enumerate(Xg):
        if np.isclose(t, t0):
            # This is a boundary point - mask it out.
            mask[i] = False

    # Flatten the mask.
    mask.shape = (ng_total,)

    # Extract the internal points.
    Xg_in = Xg[mask]

    # Invert the mask and extract the boundary points.
    mask = np.logical_not(mask)
    Xg_bc = Xg[mask]

    # Return the complete, inner, and boundary training points.
    return Xg, Xg_in, Xg_bc


def compute_boundary_conditions(X: np.ndarray):
    """Compute the boundary conditions.

    The boundary conditions are computed using the analytical solutions at the
    specified points. Note that this can be used to compute synthetic data
    values within the domain.

    Parameters
    ----------
    X : np.ndarray of float, shape (n_bc, n_dim)
        Independent variable values for computation.

    Returns
    -------
    bc : np.ndarray of float, shape (n_bc, n_var)
        Values of each dependent variable at X.
    """
    n = len(X)
    bc = np.empty((n, n_var))
    for (i, ψa) in enumerate(ψ_analytical):
        bc[:, i] = ψa(X)
    return bc


# @tf.function
def pde_ρ1(X, ψ, del_ψ):
    """Differential equation for mass density perturbation.

    Evaluate the differential equation for mass density perturbation. This
    equation is derived from the equation of mass continuity.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    ψ : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    del_ψ : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables at
        each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.
    """
    n = X.shape[0]
    t = tf.reshape(X[:, 0], (n, 1))
    x = tf.reshape(X[:, 1], (n, 1))
    (ρ1, P1, u1x, u1y, u1z, B1x, B1y, B1z) = ψ
    (del_ρ1, del_P1, del_u1x, del_u1y, del_u1z, del_B1x, del_B1y, del_B1z) = del_ψ
    dρ1_dt = tf.reshape(del_ρ1[:, 0], (n, 1))
    dρ1_dx = tf.reshape(del_ρ1[:, 1], (n, 1))
    dP1_dt = tf.reshape(del_P1[:, 0], (n, 1))
    dP1_dx = tf.reshape(del_P1[:, 1], (n, 1))
    du1x_dt = tf.reshape(del_u1x[:, 0], (n, 1))
    du1x_dx = tf.reshape(del_u1x[:, 1], (n, 1))
    du1y_dt = tf.reshape(del_u1y[:, 0], (n, 1))
    du1y_dx = tf.reshape(del_u1y[:, 1], (n, 1))
    du1z_dt = tf.reshape(del_u1z[:, 0], (n, 1))
    du1z_dx = tf.reshape(del_u1z[:, 1], (n, 1))
    dB1x_dt = tf.reshape(del_B1x[:, 0], (n, 1))
    dB1x_dx = tf.reshape(del_B1x[:, 1], (n, 1))
    dB1y_dt = tf.reshape(del_B1y[:, 0], (n, 1))
    dB1y_dx = tf.reshape(del_B1y[:, 1], (n, 1))
    dB1z_dt = tf.reshape(del_B1z[:, 0], (n, 1))
    dB1z_dx = tf.reshape(del_B1z[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = dρ1_dt + ρ0*du1x_dx
    return G


# @tf.function
def pde_P1(X, ψ, del_ψ):
    """Differential equation for pressure perturbation.

    Evaluate the differential equation for pressure perturbation. This
    equation is derived from the equation of energy continuity.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    ψ : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    del_ψ : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables at
        each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.
    """
    n = X.shape[0]
    t = tf.reshape(X[:, 0], (n, 1))
    x = tf.reshape(X[:, 1], (n, 1))
    (ρ1, P1, u1x, u1y, u1z, B1x, B1y, B1z) = ψ
    (del_ρ1, del_P1, del_u1x, del_u1y, del_u1z, del_B1x, del_B1y, del_B1z) = del_ψ
    dρ1_dt = tf.reshape(del_ρ1[:, 0], (n, 1))
    dρ1_dx = tf.reshape(del_ρ1[:, 1], (n, 1))
    dP1_dt = tf.reshape(del_P1[:, 0], (n, 1))
    dP1_dx = tf.reshape(del_P1[:, 1], (n, 1))
    du1x_dt = tf.reshape(del_u1x[:, 0], (n, 1))
    du1x_dx = tf.reshape(del_u1x[:, 1], (n, 1))
    du1y_dt = tf.reshape(del_u1y[:, 0], (n, 1))
    du1y_dx = tf.reshape(del_u1y[:, 1], (n, 1))
    du1z_dt = tf.reshape(del_u1z[:, 0], (n, 1))
    du1z_dx = tf.reshape(del_u1z[:, 1], (n, 1))
    dB1x_dt = tf.reshape(del_B1x[:, 0], (n, 1))
    dB1x_dx = tf.reshape(del_B1x[:, 1], (n, 1))
    dB1y_dt = tf.reshape(del_B1y[:, 0], (n, 1))
    dB1y_dx = tf.reshape(del_B1y[:, 1], (n, 1))
    dB1z_dt = tf.reshape(del_B1z[:, 0], (n, 1))
    dB1z_dx = tf.reshape(del_B1z[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = dP1_dt - ɣ*P0/ρ0*dρ1_dt
    return G


# @tf.function
def pde_u1x(X, ψ, del_ψ):
    """Differential equation for x-velocity perturbation.

    Evaluate the differential equation for x-velocity perturbation. This
    equation is derived from the equation of conservation of x-momentum.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    ψ : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    del_ψ : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables at
        each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.
    """
    n = X.shape[0]
    t = tf.reshape(X[:, 0], (n, 1))
    x = tf.reshape(X[:, 1], (n, 1))
    (ρ1, P1, u1x, u1y, u1z, B1x, B1y, B1z) = ψ
    (del_ρ1, del_P1, del_u1x, del_u1y, del_u1z, del_B1x, del_B1y, del_B1z) = del_ψ
    dρ1_dt = tf.reshape(del_ρ1[:, 0], (n, 1))
    dρ1_dx = tf.reshape(del_ρ1[:, 1], (n, 1))
    dP1_dt = tf.reshape(del_P1[:, 0], (n, 1))
    dP1_dx = tf.reshape(del_P1[:, 1], (n, 1))
    du1x_dt = tf.reshape(del_u1x[:, 0], (n, 1))
    du1x_dx = tf.reshape(del_u1x[:, 1], (n, 1))
    du1y_dt = tf.reshape(del_u1y[:, 0], (n, 1))
    du1y_dx = tf.reshape(del_u1y[:, 1], (n, 1))
    du1z_dt = tf.reshape(del_u1z[:, 0], (n, 1))
    du1z_dx = tf.reshape(del_u1z[:, 1], (n, 1))
    dB1x_dt = tf.reshape(del_B1x[:, 0], (n, 1))
    dB1x_dx = tf.reshape(del_B1x[:, 1], (n, 1))
    dB1y_dt = tf.reshape(del_B1y[:, 0], (n, 1))
    dB1y_dx = tf.reshape(del_B1y[:, 1], (n, 1))
    dB1z_dt = tf.reshape(del_B1z[:, 0], (n, 1))
    dB1z_dx = tf.reshape(del_B1z[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = ρ0*du1x_dt + dP1_dx
    return G


# @tf.function
def pde_u1y(X, ψ, del_ψ):
    """Differential equation for y-velocity perturbation.

    Evaluate the differential equation for y-velocity perturbation. This equation
    is derived from the equation of conservation of y-momentum.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    ψ : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    del_ψ : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables at
        each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.
    """
    n = X.shape[0]
    t = tf.reshape(X[:, 0], (n, 1))
    x = tf.reshape(X[:, 1], (n, 1))
    (ρ1, P1, u1x, u1y, u1z, B1x, B1y, B1z) = ψ
    (del_ρ1, del_P1, del_u1x, del_u1y, del_u1z, del_B1x, del_B1y, del_B1z) = del_ψ
    dρ1_dt = tf.reshape(del_ρ1[:, 0], (n, 1))
    dρ1_dx = tf.reshape(del_ρ1[:, 1], (n, 1))
    dP1_dt = tf.reshape(del_P1[:, 0], (n, 1))
    dP1_dx = tf.reshape(del_P1[:, 1], (n, 1))
    du1x_dt = tf.reshape(del_u1x[:, 0], (n, 1))
    du1x_dx = tf.reshape(del_u1x[:, 1], (n, 1))
    du1y_dt = tf.reshape(del_u1y[:, 0], (n, 1))
    du1y_dx = tf.reshape(del_u1y[:, 1], (n, 1))
    du1z_dt = tf.reshape(del_u1z[:, 0], (n, 1))
    du1z_dx = tf.reshape(del_u1z[:, 1], (n, 1))
    dB1x_dt = tf.reshape(del_B1x[:, 0], (n, 1))
    dB1x_dx = tf.reshape(del_B1x[:, 1], (n, 1))
    dB1y_dt = tf.reshape(del_B1y[:, 0], (n, 1))
    dB1y_dx = tf.reshape(del_B1y[:, 1], (n, 1))
    dB1z_dt = tf.reshape(del_B1z[:, 0], (n, 1))
    dB1z_dx = tf.reshape(del_B1z[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = ρ0*du1y_dt - B0x*dB1y_dx/μ0
    return G


# @tf.function
def pde_u1z(X, ψ, del_ψ):
    """Differential equation for z-velocity perturbation.

    Evaluate the differential equation for z-velocity perturbation.
    This equation is derived from the equation of conservation of z-momentum.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    ψ : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    del_ψ : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables at
        each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.
    """
    n = X.shape[0]
    t = tf.reshape(X[:, 0], (n, 1))
    x = tf.reshape(X[:, 1], (n, 1))
    (ρ1, P1, u1x, u1y, u1z, B1x, B1y, B1z) = ψ
    (del_ρ1, del_P1, del_u1x, del_u1y, del_u1z, del_B1x, del_B1y, del_B1z) = del_ψ
    dρ1_dt = tf.reshape(del_ρ1[:, 0], (n, 1))
    dρ1_dx = tf.reshape(del_ρ1[:, 1], (n, 1))
    dP1_dt = tf.reshape(del_P1[:, 0], (n, 1))
    dP1_dx = tf.reshape(del_P1[:, 1], (n, 1))
    du1x_dt = tf.reshape(del_u1x[:, 0], (n, 1))
    du1x_dx = tf.reshape(del_u1x[:, 1], (n, 1))
    du1y_dt = tf.reshape(del_u1y[:, 0], (n, 1))
    du1y_dx = tf.reshape(del_u1y[:, 1], (n, 1))
    du1z_dt = tf.reshape(del_u1z[:, 0], (n, 1))
    du1z_dx = tf.reshape(del_u1z[:, 1], (n, 1))
    dB1x_dt = tf.reshape(del_B1x[:, 0], (n, 1))
    dB1x_dx = tf.reshape(del_B1x[:, 1], (n, 1))
    dB1y_dt = tf.reshape(del_B1y[:, 0], (n, 1))
    dB1y_dx = tf.reshape(del_B1y[:, 1], (n, 1))
    dB1z_dt = tf.reshape(del_B1z[:, 0], (n, 1))
    dB1z_dx = tf.reshape(del_B1z[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = ρ0*du1z_dt - B0x*dB1z_dx/μ0
    return G


# @tf.function
def pde_B1x(X, ψ, del_ψ):
    """Differential equation for x-magnetic field perturbation.

    Evaluate the differential equation for x-magnetic field perturbation.
    This equation is derived from the x-component of Faraday's Law.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    ψ : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    del_ψ : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables at
        each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.
    """
    n = X.shape[0]
    t = tf.reshape(X[:, 0], (n, 1))
    x = tf.reshape(X[:, 1], (n, 1))
    (ρ1, P1, u1x, u1y, u1z, B1x, B1y, B1z) = ψ
    (del_ρ1, del_P1, del_u1x, del_u1y, del_u1z, del_B1x, del_B1y, del_B1z) = del_ψ
    dρ1_dt = tf.reshape(del_ρ1[:, 0], (n, 1))
    dρ1_dx = tf.reshape(del_ρ1[:, 1], (n, 1))
    dP1_dt = tf.reshape(del_P1[:, 0], (n, 1))
    dP1_dx = tf.reshape(del_P1[:, 1], (n, 1))
    du1x_dt = tf.reshape(del_u1x[:, 0], (n, 1))
    du1x_dx = tf.reshape(del_u1x[:, 1], (n, 1))
    du1y_dt = tf.reshape(del_u1y[:, 0], (n, 1))
    du1y_dx = tf.reshape(del_u1y[:, 1], (n, 1))
    du1z_dt = tf.reshape(del_u1z[:, 0], (n, 1))
    du1z_dx = tf.reshape(del_u1z[:, 1], (n, 1))
    dB1x_dt = tf.reshape(del_B1x[:, 0], (n, 1))
    dB1x_dx = tf.reshape(del_B1x[:, 1], (n, 1))
    dB1y_dt = tf.reshape(del_B1y[:, 0], (n, 1))
    dB1y_dx = tf.reshape(del_B1y[:, 1], (n, 1))
    dB1z_dt = tf.reshape(del_B1z[:, 0], (n, 1))
    dB1z_dx = tf.reshape(del_B1z[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = tf.zeros((n,))
    return G


# @tf.function
def pde_B1y(X, ψ, del_ψ):
    """Differential equation for y-magnetic field perturbation.

    Evaluate the differential equation for y-magnetic field perturbation.
    This equation is derived from the y-component of Faraday's Law.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    ψ : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    del_ψ : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables at
        each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.
    """
    n = X.shape[0]
    t = tf.reshape(X[:, 0], (n, 1))
    x = tf.reshape(X[:, 1], (n, 1))
    (ρ1, P1, u1x, u1y, u1z, B1x, B1y, B1z) = ψ
    (del_ρ1, del_P1, del_u1x, del_u1y, del_u1z, del_B1x, del_B1y, del_B1z) = del_ψ
    dρ1_dt = tf.reshape(del_ρ1[:, 0], (n, 1))
    dρ1_dx = tf.reshape(del_ρ1[:, 1], (n, 1))
    dP1_dt = tf.reshape(del_P1[:, 0], (n, 1))
    dP1_dx = tf.reshape(del_P1[:, 1], (n, 1))
    du1x_dt = tf.reshape(del_u1x[:, 0], (n, 1))
    du1x_dx = tf.reshape(del_u1x[:, 1], (n, 1))
    du1y_dt = tf.reshape(del_u1y[:, 0], (n, 1))
    du1y_dx = tf.reshape(del_u1y[:, 1], (n, 1))
    du1z_dt = tf.reshape(del_u1z[:, 0], (n, 1))
    du1z_dx = tf.reshape(del_u1z[:, 1], (n, 1))
    dB1x_dt = tf.reshape(del_B1x[:, 0], (n, 1))
    dB1x_dx = tf.reshape(del_B1x[:, 1], (n, 1))
    dB1y_dt = tf.reshape(del_B1y[:, 0], (n, 1))
    dB1y_dx = tf.reshape(del_B1y[:, 1], (n, 1))
    dB1z_dt = tf.reshape(del_B1z[:, 0], (n, 1))
    dB1z_dx = tf.reshape(del_B1z[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = dB1y_dt - B0x*du1y_dx
    return G


# @tf.function
def pde_B1z(X, ψ, del_ψ):
    """Differential equation for z-magnetic field perturbation.

    Evaluate the differential equation for z-magnetic field perturbation.
    This equation is derived from the z-component of Faraday's Law.

    Parameters
    ----------
    X : tf.Variable, shape (n, n_dim)
        Values of independent variables at each evaluation point.
    ψ : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each evaluation point.
    del_ψ : list of n_var tf.Tensor, each shape (n, n_dim)
        Values of gradients of dependent variables wrt independent variables at
        each evaluation point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of differential equation at each evaluation point.
    """
    n = X.shape[0]
    t = tf.reshape(X[:, 0], (n, 1))
    x = tf.reshape(X[:, 1], (n, 1))
    (ρ1, P1, u1x, u1y, u1z, B1x, B1y, B1z) = ψ
    (del_ρ1, del_P1, del_u1x, del_u1y, del_u1z, del_B1x, del_B1y, del_B1z) = del_ψ
    dρ1_dt = tf.reshape(del_ρ1[:, 0], (n, 1))
    dρ1_dx = tf.reshape(del_ρ1[:, 1], (n, 1))
    dP1_dt = tf.reshape(del_P1[:, 0], (n, 1))
    dP1_dx = tf.reshape(del_P1[:, 1], (n, 1))
    du1x_dt = tf.reshape(del_u1x[:, 0], (n, 1))
    du1x_dx = tf.reshape(del_u1x[:, 1], (n, 1))
    du1y_dt = tf.reshape(del_u1y[:, 0], (n, 1))
    du1y_dx = tf.reshape(del_u1y[:, 1], (n, 1))
    du1z_dt = tf.reshape(del_u1z[:, 0], (n, 1))
    du1z_dx = tf.reshape(del_u1z[:, 1], (n, 1))
    dB1x_dt = tf.reshape(del_B1x[:, 0], (n, 1))
    dB1x_dx = tf.reshape(del_B1x[:, 1], (n, 1))
    dB1y_dt = tf.reshape(del_B1y[:, 0], (n, 1))
    dB1y_dx = tf.reshape(del_B1y[:, 1], (n, 1))
    dB1z_dt = tf.reshape(del_B1z[:, 0], (n, 1))
    dB1z_dx = tf.reshape(del_B1z[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = dB1z_dt - B0x*du1z_dx
    return G


# Make a list of all of the differential equations.
differential_equations = [
    pde_ρ1,
    pde_P1,
    pde_u1x,
    pde_u1y,
    pde_u1z,
    pde_B1x,
    pde_B1y,
    pde_B1z
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

    print("ρ0 = %s" % ρ0)
    print("P0 = %s" % P0)
    print("u0x = %s" % u0x)
    print("u0y = %s" % u0y)
    print("u0z = %s" % u0z)
    print("B0x = %s" % B0x)
    print("B0y = %s" % B0y)
    print("B0z = %s" % B0z)

    print("ρ1_amp = %s" % ρ1_amp)
    print("P1_amp = %s" % P1_amp)
    print("u1x_amp = %s" % u1x_amp)
    print("u1y_amp = %s" % u1y_amp)
    print("u1z_amp = %s" % u1z_amp)
    print("B1x_amp = %s" % B1x_amp)
    print("B1y_amp = %s" % B1y_amp)
    print("B1z_amp = %s" % B1z_amp)

    nt = 3
    nx = 4
    Xg, Xg_in, Xg_bc = create_training_data_gridded(nt, nx)
    print("Xg = %s" % Xg)
    print("Xg_in = %s" % Xg_in)
    print("Xg_bc = %s" % Xg_bc)

    bc = compute_boundary_conditions(Xg_bc)
    print("bc = %s" % bc)
