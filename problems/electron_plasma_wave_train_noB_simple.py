"""Problem definition file for a simple 1-D MHD problem.

This problem definition file describes an electron plasma wave, using
the reduced set of MHD equations (3 variables, 3 PDEs).  Note that
these equations use the electron number density (n), rather than the
mass density (ρ).

This problem specifies the initial conditions, and the boundary conditions at
x = 0, so the result is a wave train propagating in t and x.

NOTE: The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by the TensorFlow
code.

NOTE: In all code, below, the following indices are assigned to physical
independent variables:

    0: t
    1: x

NOTE: In all code, below, the following indices are assigned to physical
dependent variables:

     0: n1
     1: u1x
     2: E1x

Author
------
Eric Winter (eric.winter62@gmail.com)

"""


# Import standard modules.

# Import supplemental modules.
import numpy as np
import tensorflow as tf

# Import project modules.
import nn1dmhd.plasma as plasma
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
dependent_variable_names = ['n1', 'u1x', 'E1x']

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = [r'$n_1$', r'$u_{1x}$', r'$E_{1x}$']

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
n0 = 1.0
u0x = 0.0
E0x = 0.0
initial_values = [n0, u0x, E0x]


# Plasma computed values

# Compute the electron thermal speed (independent of components).
vth = plasma.electron_thermal_speed(T, normalize=True)

# Compute the electron plasma angular frequency (independent of components).
ωp = plasma.electron_plasma_angular_frequency(n0*𝑚e, normalize=True)

# Perturbations

# Wavelength and wavenumber of initial density/velocity/Ex perturbation.
λ = 1.0
kx = 2*np.pi/λ

# Compute the electron plasma wave angular frequency.
ω = plasma.electron_plasma_wave_angular_frequency(n0*𝑚e, T, kx, normalize=True)

# Compute the wave phase speed.
vφ = plasma.electron_plasma_wave_phase_speed(n0*𝑚e, T, kx, normalize=True)

# Perturbation amplitudes for each dependent variable (dimensionless).
n1_amp = 0.1
u1x_amp = ω*n1_amp/(kx*n0)
E1x_amp = 𝑒*n1_amp/(kx*ε0)


def n1_analytical(X: np.ndarray):
    """Compute analytical solution for number density perturbation.

    Compute analytical solution for number density perturbation.

    Parameters
    ----------
    X : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    n1 : np.ndarray of float, shape (n,)
        Analytical values for number density perturbation.
    """
    t = X[:, 0]
    x = X[:, 1]
    n1 = n1_amp*np.sin(kx*x - ω*t)
    return n1


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
    t = X[:, 0]
    x = X[:, 1]
    u1x = u1x_amp*np.sin(kx*x - ω*t)
    return u1x


def E1x_analytical(X: np.ndarray):
    """Compute analytical solution for x-electric field perturbation.

    Compute anaytical solution for x-electric field perturbation.

    Parameters
    ----------
    X : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    E1x : np.ndarray of float, shape (n,)
        Analytical values for x-electric field perturbation.
    """
    t = X[:, 0]
    x = X[:, 1]
    E1x = E1x_amp*np.sin(kx*x - ω*t + np.pi/2)
    return E1x


# Gather all of the analytical solutions into a list.
ψ_analytical = [
    n1_analytical,
    u1x_analytical,
    E1x_analytical,
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
        if np.isclose(t, t0) or np.isclose(x, x0):
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
def pde_n1(X, ψ, del_ψ):
    """Differential equation for number density perturbation.

    Evaluate the differential equation for number density perturbation. This
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
    (n1, u1x, E1x) = ψ
    (del_n1, del_u1x, del_E1x) = del_ψ
    dn1_dt = tf.reshape(del_n1[:, 0], (n, 1))
    dn1_dx = tf.reshape(del_n1[:, 1], (n, 1))
    du1x_dt = tf.reshape(del_u1x[:, 0], (n, 1))
    du1x_dx = tf.reshape(del_u1x[:, 1], (n, 1))
    dE1x_dt = tf.reshape(del_E1x[:, 0], (n, 1))
    dE1x_dx = tf.reshape(del_E1x[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = dn1_dt + n0*du1x_dx
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
    (n1, u1x, E1x) = ψ
    (del_n1, del_u1x, del_E1x) = del_ψ
    dn1_dt = tf.reshape(del_n1[:, 0], (n, 1))
    dn1_dx = tf.reshape(del_n1[:, 1], (n, 1))
    du1x_dt = tf.reshape(del_u1x[:, 0], (n, 1))
    du1x_dx = tf.reshape(del_u1x[:, 1], (n, 1))
    dE1x_dt = tf.reshape(del_E1x[:, 0], (n, 1))
    dE1x_dx = tf.reshape(del_E1x[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = du1x_dt + ɣ*𝑘b*T/(𝑚e*n0)*dn1_dx + 𝑒/𝑚e*E1x
    return G


# @tf.function
def pde_E1x(X, ψ, del_ψ):
    """Differential equation for x-electric field perturbation.

    Evaluate the differential equation for x-electric field perturbation.
    This equation is derived from the x-component of Ampere's Law.

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
    (n1, u1x, E1x) = ψ
    (del_n1, del_u1x, del_E1x) = del_ψ
    dn1_dt = tf.reshape(del_n1[:, 0], (n, 1))
    dn1_dx = tf.reshape(del_n1[:, 1], (n, 1))
    du1x_dt = tf.reshape(del_u1x[:, 0], (n, 1))
    du1x_dx = tf.reshape(del_u1x[:, 1], (n, 1))
    dE1x_dt = tf.reshape(del_E1x[:, 0], (n, 1))
    dE1x_dx = tf.reshape(del_E1x[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = dE1x_dx - 𝑒/ε0*n1
    return G


# Make a list of all of the differential equations.
differential_equations = [
    pde_n1,
    pde_u1x,
    pde_E1x,
]


if __name__ == "__main__":
    print("independent_variable_names = %s" % independent_variable_names)
    print("independent_variable_labels = %s" % independent_variable_labels)
    # print("n_dim = %s" % n_dim)
    # print("dependent_variable_names = %s" % dependent_variable_names)
    # print("dependent_variable_labels = %s" % dependent_variable_labels)
    # print("n_var = %s" % n_var)

    # print("%s <= t <= %s" % (t0, t1))
    # print("%s <= x <= %s" % (x0, x1))
    # print("domain = %s" % domain)

    # print("n0 = %s" % n0)
    # print("P0 = %s" % P0)
    # print("u0x = %s" % u0x)
    # print("u0y = %s" % u0y)
    # print("u0z = %s" % u0z)
    # print("E0x = %s" % E0x)
    # print("E0y = %s" % E0y)
    # print("E0z = %s" % E0z)

    # print("n1_amp = %s" % n1_amp)
    # print("P1_amp = %s" % P1_amp)
    # print("u1x_amp = %s" % u1x_amp)
    # print("u1y_amp = %s" % u1y_amp)
    # print("u1z_amp = %s" % u1z_amp)
    # print("E1x_amp = %s" % E1x_amp)
    # print("E1y_amp = %s" % E1y_amp)
    # print("E1z_amp = %s" % E1z_amp)

    # nt = 3
    # nx = 4
    # Xg, Xg_in, Xg_bc = create_training_data_gridded(nt, nx)
    # print("Xg = %s" % Xg)
    # print("Xg_in = %s" % Xg_in)
    # print("Xg_bc = %s" % Xg_bc)

    # bc = compute_boundary_conditions(Xg_bc)
    # print("bc = %s" % bc)
