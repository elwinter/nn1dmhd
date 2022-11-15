"""Problem definition file for a simple 1-D MHD problem.

This problem definition file describes an Alfven wave: unit pressure and
density, with a constant axial magnetic field (B0x = constant).

NOTE: The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by the TensorFlow
code.

NOTE: In all code, below, the following indices are assigned to physical
independent variables:

    0: t
    1: x

NOTE: In all code, below, the following indices are assigned to physical
dependent variables:

    0: ρ
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
from nn1dmhd import plasma
from nn1dmhd.training_data import create_training_points_gridded


# Normalized physical constants.
𝑒 = 1.0     # Unit charge
𝑘b = 1.0    # Boltzmann constant
ε0 = 1.0  # Permeability of free space
𝑚e = 1.0    # Electron mass

# Names of independent variables.
independent_variable_names = ['t', 'x']

# Labels for independent variables (may use LaTex) - use for plots.
independent_variable_labels = [r'$t$', r'$x$']

# Number of problem dimensions (independent variables).
n_dim = len(independent_variable_names)

# Names of dependent variables.
dependent_variable_names = ['ρ', 'P', 'ux', 'uy', 'uz', 'Bx', 'By', 'Bz']

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = [
    r'$\rho$', r'$P$',
    r'$u_x$', r'$u_y$', r'$u_z$',
    r'$B_x$', r'$B_y$', r'$B_z$'
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
ux0 = 0.0
uy0 = 0.0
uz0 = 0.0
Bx0 = 0.0
By0 = 0.0
Bz0 = 0.0
initial_values = [ρ0, P0, ux0, uy0, uz0, Bx0, By0, Bz0]


# Plasma computed values

# Compute the electron thermal speed (independent of components).
vth = plasma.electron_thermal_speed(T, normalize=True)

# Compute the electron plasma angular frequency (independent of components).
ωp = plasma.electron_plasma_angular_frequency(ρ0, normalize=True)

# Alfven speed.
C_Alfven = Bx0/np.sqrt(ρ0)


# Perturbations

# Perturbation amplitudes for each dependent variable (dimensionless).
ρ1_amp = 0.0
P1_amp = 0.0
ux1_amp = 0.0
uy1_amp = 0.0
uz1_amp = 0.0
Bx1_amp = 0.0
By1_amp = 0.0
Bz1_amp = 0.0

# Wavelength and wavenumber of initial perturbations.
λ = np.array([1.0])
nc = len(λ)  # Number of wave components.
kx = 2*np.pi/λ

# Compute the electron plasma wave angular frequency for each component.                           
ω = plasma.electron_plasma_wave_angular_frequency(ρ0, T, kx, normalize=True)

# Compute the wave phase speed for each component.
vphase = plasma.electron_plasma_wave_phase_speed(ρ0, T, kx, normalize=True)

# Frequency and angular frequency of initial perturbation.
f = C_Alfven/λ
# w = 2*np.pi*f


def ρ_analytical(xt: np.ndarray):
    """Compute analytical solution for mass density.

    Compute anaytical solution for mass density.

    Parameters
    ----------
    xt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    ρ : np.ndarray of float, shape (n,)
        Analytical values for mass density.
    """
    ρ = np.full((xt.shape[0],), ρ0)
    return ρ


def P_analytical(xt: np.ndarray):
    """Compute analytical solution for pressure.

    Compute anaytical solution for pressure.

    Parameters
    ----------
    xt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    P : np.ndarray of float, shape (n,)
        Analytical values for pressure.
    """
    P = np.full((xt.shape[0],), P0)
    return P


def ux_analytical(xt: np.ndarray):
    """Compute analytical solution for x-velocity.

    Compute anaytical solution for x-velocity.

    Parameters
    ----------
    xt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    ux : np.ndarray of float, shape (n,)
        Analytical values for x-velocity.
    """
    ux = np.full((xt.shape[0],), ux0)
    return ux


def uy_analytical(xt: np.ndarray):
    """Compute analytical solution for y-velocity.

    Compute anaytical solution for y-velocity.

    Parameters
    ----------
    xt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    uy : np.ndarray of float, shape (n,)
        Analytical values for y-velocity.
    """
    uy = np.full((xt.shape[0],), uy0)
    return uy


def uz_analytical(xt: np.ndarray):
    """Compute analytical solution for z-velocity.

    Compute anaytical solution for z-velocity.

    Parameters
    ----------
    xt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    uz : np.ndarray of float, shape (n,)
        Analytical values for z-velocity.
    """
    uz = np.full((xt.shape[0],), uz0)
    return uz


def Bx_analytical(xt: np.ndarray):
    """Compute analytical solution for x-magnetic field.

    Compute anaytical solution for x-magnetic field.

    Parameters
    ----------
    xt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    Bx : np.ndarray of float, shape (n,)
        Analytical values for x-magnetic field.
    """
    Bx = np.full((xt.shape[0],), Bx0)
    return Bx


def By_analytical(xt: np.ndarray):
    """Compute analytical solution for y-magnetic field.

    Compute anaytical solution for y-magnetic field.

    Parameters
    ----------
    xt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    By : np.ndarray of float, shape (n,)
        Analytical values for y-magnetic field.
    """
    By = np.full((xt.shape[0],), By0)
    return By


def Bz_analytical(xt: np.ndarray):
    """Compute analytical solution for z-magnetic field.

    Compute anaytical solution for z-magnetic field.

    Parameters
    ----------
    xt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    Bz : np.ndarray of float, shape (n,)
        Analytical values for z-magnetic field.
    """
    Bz = np.full((xt.shape[0],), Bz0)
    return Bz


# Gather all of the analytical solutions into a list.
ψ_analytical = [
    ρ_analytical,
    P_analytical,
    ux_analytical,
    uy_analytical,
    uz_analytical,
    Bx_analytical,
    By_analytical,
    Bz_analytical,
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


# def compute_boundary_conditions(xt: np.ndarray):
#     """Compute the boundary conditions.

#     The boundary conditions are computed using the analytical solutions at the
#     specified points. Note that this can be used to compute synthetic data
#     values within the domain.

#     Parameters
#     ----------
#     xt : np.ndarray of float, shape (n_bc, n_dim)
#         Independent variable values for computation.

#     Returns
#     -------
#     bc : np.ndarray of float, shape (n_bc, n_var)
#         Values of each dependent variable at xt.
#     """
#     n = len(xt)
#     bc = np.empty((n, n_var))
#     for (i, ψa) in enumerate(ψ_analytical):
#         bc[:, i] = ψa(xt)
#     return bc


# # Define the differential equations using TensorFlow operations.

# # @tf.function
# def pde_ρ1(xt, ψ1, del_ψ1):
#     """Differential equation for rho1.

#     Evaluate the differential equation for rho1 (density perturbation).

#     Parameters
#     ----------
#     xt : tf.Variable, shape (n, 2)
#         Values of (x, t) at each training point.
#     ψ1 : list of n_var tf.Tensor, each shape (n, 1)
#         Perturbations of dependent variables at each training point.
#     del_ψ1 : list of n_var tf.Tensor, each shape (n, 2)
#         Values of gradients of ψ1 wrt (x, t) at each training point.

#     Returns
#     -------
#     G : tf.Tensor, shape(n, 1)
#         Value of differential equation at each training point.
#     """
#     # Each of these Tensors is shape (n, 1).
#     n = xt.shape[0]
#     # x = tf.reshape(xt[:, 0], (n, 1))
#     # t = tf.reshape(xt[:, 1], (n, 1))
#     (rho1, v1x, v1y, v1z, B1y, B1z) = Y1
#     (del_rho1, del_v1x, del_v1y, del_v1z, del_B1y, del_B1z) = del_Y1
#     # drho1_dx = tf.reshape(del_rho1[:, 0], (n, 1))
#     drho1_dt = tf.reshape(del_rho1[:, 1], (n, 1))
#     dv1x_dx = tf.reshape(del_v1x[:, 0], (n, 1))
#     # dv1x_dt = tf.reshape(del_v1x[:, 1], (n, 1))
#     # dv1y_dx = tf.reshape(del_v1y[:, 0], (n, 1))
#     # dv1y_dt = tf.reshape(del_v1y[:, 1], (n, 1))
#     # dv1z_dx = tf.reshape(del_v1z[:, 0], (n, 1))
#     # dv1z_dt = tf.reshape(del_v1z[:, 1], (n, 1))
#     # dB1y_dx = tf.reshape(del_B1y[:, 0], (n, 1))
#     # dB1y_dt = tf.reshape(del_B1y[:, 1], (n, 1))
#     # dB1z_dx = tf.reshape(del_B1z[:, 0], (n, 1))
#     # dB1z_dt = tf.reshape(del_B1z[:, 1], (n, 1))

#     # G is a Tensor of shape (n, 1).
#     G = drho1_dt + rho0*dv1x_dx
#     return G


# # @tf.function
# def pde_P1(xt, ψ1, del_ψ1):
#     """Differential equation for P1.

#     Evaluate the differential equation for pressure perturbation P1.
#     This equation is derived from the equation of conservation of energy.

#     Parameters
#     ----------
#     xt : tf.Variable, shape (n, n_dim)
#         Values of independent variables at each evaluation point.
#     ψ : list of n_var tf.Tensor, each shape (n, 1)
#         Values of dependent variables at each evaluation point.
#     del_ψ : list of n_var tf.Tensor, each shape (n, n_dim)
#         Values of gradients of dependent variables wrt independent variables at
#         each evaluation point.

#     Returns
#     -------
#     G : tf.Tensor, shape (n, 1)
#         Value of differential equation at each evaluation point.
#     """
#     n = xt.shape[0]
#     # x = tf.reshape(xt[:, 0], (n, 1))
#     # t = tf.reshape(xt[:, 1], (n, 1))
#     (ρ, P, ux, uy, uz, Bx, By, Bz) = ψ
#     (del_ρ, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_ψ
#     dρ_dx = tf.reshape(del_ρ[:, 0], (n, 1))
#     dρ_dt = tf.reshape(del_ρ[:, 1], (n, 1))
#     dP_dx = tf.reshape(del_P[:, 0], (n, 1))
#     dP_dt = tf.reshape(del_P[:, 1], (n, 1))
#     # dux_dx = tf.reshape(del_ux[:, 0], (n, 1))
#     # dux_dt = tf.reshape(del_ux[:, 1], (n, 1))
#     # duy_dx = tf.reshape(del_uy[:, 0], (n, 1))
#     # duy_dt = tf.reshape(del_uy[:, 1], (n, 1))
#     # duz_dx = tf.reshape(del_uz[:, 0], (n, 1))
#     # duz_dt = tf.reshape(del_uz[:, 1], (n, 1))
#     # dBx_dx = tf.reshape(del_Bx[:, 0], (n, 1))
#     # dBx_dt = tf.reshape(del_Bx[:, 1], (n, 1))
#     # dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
#     # dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
#     # dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
#     # dBz_dt = tf.reshape(del_Bz[:, 1], (n, 1))

#     # G is a Tensor of shape (n, 1).
#     G = -gamma*P/ρ*(dρ_dt + ux*dρ_dx) + dP_dt + ux*dP_dx
#     return G


# # @tf.function
# def pde_v1x(xt, Y1, del_Y1):
#     """Differential equation for v1x.

#     Evaluate the differential equation for v1x (x-component of velocity
#     perturbation).

#     Parameters
#     ----------
#     xt : tf.Variable, shape (n, 2)
#         Values of (x, t) at each training point.
#     Y1 : list of n_var tf.Tensor, each shape (n, 1)
#         Perturbations of dependent variables at each training point.
#     del_Y1 : list of n_var tf.Tensor, each shape (n, 2)
#         Values of gradients of Y1 wrt (x, t) at each training point.

#     Returns
#     -------
#     G : tf.Tensor, shape(n, 1)
#         Value of differential equation at each training point.
#     """
#     # Each of these Tensors is shape (n, 1).
#     n = xt.shape[0]
#     # x = tf.reshape(xt[:, 0], (n, 1))
#     # t = tf.reshape(xt[:, 1], (n, 1))
#     # (rho1, v1x, v1y, v1z, B1y, B1z) = Y1
#     (del_rho1, del_v1x, del_v1y, del_v1z, del_B1y, del_B1z) = del_Y1
#     drho1_dx = tf.reshape(del_rho1[:, 0], (n, 1))
#     # drho1_dt = tf.reshape(del_rho1[:, 1], (n, 1))
#     # dv1x_dx = tf.reshape(del_v1x[:, 0], (n, 1))
#     dv1x_dt = tf.reshape(del_v1x[:, 1], (n, 1))
#     # dv1y_dx = tf.reshape(del_v1y[:, 0], (n, 1))
#     # dv1y_dt = tf.reshape(del_v1y[:, 1], (n, 1))
#     # dv1z_dx = tf.reshape(del_v1z[:, 0], (n, 1))
#     # dv1z_dt = tf.reshape(del_v1z[:, 1], (n, 1))
#     # dB1y_dx = tf.reshape(del_B1y[:, 0], (n, 1))
#     # dB1y_dt = tf.reshape(del_B1y[:, 1], (n, 1))
#     # dB1z_dx = tf.reshape(del_B1z[:, 0], (n, 1))
#     # dB1z_dt = tf.reshape(del_B1z[:, 1], (n, 1))

#     # Compute the presure derivative from the density derivative.
#     dP1_dx = gamma*P0/rho0*drho1_dx

#     # G is a Tensor of shape (n, 1).
#     G = rho0*dv1x_dt + dP1_dx
#     return G


# # @tf.function
# def pde_v1y(xt, Y1, del_Y1):
#     """Differential equation for v1y.

#     Evaluate the differential equation for v1y (y-component of velocity
#     perturbation).

#     Parameters
#     ----------
#     xt : tf.Variable, shape (n, 2)
#         Values of (x, t) at each training point.
#     Y1 : list of n_var tf.Tensor, each shape (n, 1)
#         Perturbations of dependent variables at each training point.
#     del_Y1 : list of n_var tf.Tensor, each shape (n, 2)
#         Values of gradients of Y1 wrt (x, t) at each training point.

#     Returns
#     -------
#     G : tf.Tensor, shape(n, 1)
#         Value of differential equation at each training point.
#     """
#     # Each of these Tensors is shape (n, 1).
#     n = xt.shape[0]
#     # x = tf.reshape(xt[:, 0], (n, 1))
#     # t = tf.reshape(xt[:, 1], (n, 1))
#     # (rho1, v1x, v1y, v1z, B1y, B1z) = Y1
#     (del_rho1, del_v1x, del_v1y, del_v1z, del_B1y, del_B1z) = del_Y1
# #    (del_v1y, del_B1y) = del_Y1
#     # drho1_dx = tf.reshape(del_rho1[:, 0], (n, 1))
#     # drho1_dt = tf.reshape(del_rho1[:, 1], (n, 1))
#     # dv1x_dx = tf.reshape(del_v1x[:, 0], (n, 1))
#     # dv1x_dt = tf.reshape(del_v1x[:, 1], (n, 1))
#     # dv1y_dx = tf.reshape(del_v1y[:, 0], (n, 1))
#     dv1y_dt = tf.reshape(del_v1y[:, 1], (n, 1))
#     # dv1z_dx = tf.reshape(del_v1z[:, 0], (n, 1))
#     # dv1z_dt = tf.reshape(del_v1z[:, 1], (n, 1))
#     dB1y_dx = tf.reshape(del_B1y[:, 0], (n, 1))
#     # dB1y_dt = tf.reshape(del_B1y[:, 1], (n, 1))
#     # dB1z_dx = tf.reshape(del_B1z[:, 0], (n, 1))
#     # dB1z_dt = tf.reshape(del_B1z[:, 1], (n, 1))

#     # G is a Tensor of shape (n, 1).
#     G = rho0*dv1y_dt - B0x*dB1y_dx
#     return G


# # @tf.function
# def pde_v1z(xt, Y1, del_Y1):
#     """Differential equation for v1z.

#     Evaluate the differential equation for v1z (z-component of velocity
#     perturbation).

#     Parameters
#     ----------
#     xt : tf.Variable, shape (n, 2)
#         Values of (x, t) at each training point.
#     Y1 : list of n_var tf.Tensor, each shape (n, 1)
#         Perturbations of dependent variables at each training point.
#     del_Y1 : list of n_var tf.Tensor, each shape (n, 2)
#         Values of gradients of Y1 wrt (x, t) at each training point.

#     Returns
#     -------
#     G : tf.Tensor, shape(n, 1)
#         Value of differential equation at each training point.
#     """
#     # Each of these Tensors is shape (n, 1).
#     n = xt.shape[0]
#     # x = tf.reshape(xt[:, 0], (n, 1))
#     # t = tf.reshape(xt[:, 1], (n, 1))
#     # (rho1, v1x, v1y, v1z, B1y, B1z) = Y1
#     (del_rho1, del_v1x, del_v1y, del_v1z, del_B1y, del_B1z) = del_Y1
#     # drho1_dx = tf.reshape(del_rho1[:, 0], (n, 1))
#     # drho1_dt = tf.reshape(del_rho1[:, 1], (n, 1))
#     # dv1x_dx = tf.reshape(del_v1x[:, 0], (n, 1))
#     # dv1x_dt = tf.reshape(del_v1x[:, 1], (n, 1))
#     # dv1y_dx = tf.reshape(del_v1y[:, 0], (n, 1))
#     # dv1y_dt = tf.reshape(del_v1y[:, 1], (n, 1))
#     # dv1z_dx = tf.reshape(del_v1z[:, 0], (n, 1))
#     dv1z_dt = tf.reshape(del_v1z[:, 1], (n, 1))
#     # dB1y_dx = tf.reshape(del_B1y[:, 0], (n, 1))
#     # dB1y_dt = tf.reshape(del_B1y[:, 1], (n, 1))
#     dB1z_dx = tf.reshape(del_B1z[:, 0], (n, 1))
#     # dB1z_dt = tf.reshape(del_B1z[:, 1], (n, 1))

#     # G is a Tensor of shape (n, 1).
#     G = rho0*dv1z_dt - B0x*dB1z_dx
#     return G


# # @tf.function
# def pde_B1y(xt, Y1, del_Y1):
#     """Differential equation for B1y.

#     Evaluate the differential equation for B1y (y-component of magnetic
#     field perturbation).

#     Parameters
#     ----------
#     xt : tf.Variable, shape (n, 2)
#         Values of (x, t) at each training point.
#     Y1 : list of n_var tf.Tensor, each shape (n, 1)
#         Perturbations of dependent variables at each training point.
#     del_Y1 : list of n_var tf.Tensor, each shape (n, 2)
#         Values of gradients of Y1 wrt (x, t) at each training point.

#     Returns
#     -------
#     G : tf.Tensor, shape(n, 1)
#         Value of differential equation at each training point.
#     """
#     # Each of these Tensors is shape (n, 1).
#     n = xt.shape[0]
#     # x = tf.reshape(xt[:, 0], (n, 1))
#     # t = tf.reshape(xt[:, 1], (n, 1))
#     # (rho1, v1x, v1y, v1z, B1y, B1z) = Y1
#     (del_rho1, del_v1x, del_v1y, del_v1z, del_B1y, del_B1z) = del_Y1
# #    (del_v1y, del_B1y) = del_Y1
#     # drho1_dx = tf.reshape(del_rho1[:, 0], (n, 1))
#     # drho1_dt = tf.reshape(del_rho1[:, 1], (n, 1))
#     # dv1x_dx = tf.reshape(del_v1x[:, 0], (n, 1))
#     # dv1x_dt = tf.reshape(del_v1x[:, 1], (n, 1))
#     dv1y_dx = tf.reshape(del_v1y[:, 0], (n, 1))
#     # dv1y_dt = tf.reshape(del_v1y[:, 1], (n, 1))
#     # dv1z_dx = tf.reshape(del_v1z[:, 0], (n, 1))
#     # dv1z_dt = tf.reshape(del_v1z[:, 1], (n, 1))
#     # dB1y_dx = tf.reshape(del_B1y[:, 0], (n, 1))
#     dB1y_dt = tf.reshape(del_B1y[:, 1], (n, 1))
#     # dB1z_dx = tf.reshape(del_B1z[:, 0], (n, 1))
#     # dB1z_dt = tf.reshape(del_B1z[:, 1], (n, 1))

#     # G is a Tensor of shape (n, 1).
#     G = dB1y_dt - B0x*dv1y_dx
#     return G


# # @tf.function
# def pde_B1z(xt, Y1, del_Y1):
#     """Differential equation for B1z.

#     Evaluate the differential equation for B1z (z-component of magnetic
#     field perturbation).

#     Parameters
#     ----------
#     xt : tf.Variable, shape (n, 2)
#         Values of (x, t) at each training point.
#     Y1 : list of n_var tf.Tensor, each shape (n, 1)
#         Perturbations of dependent variables at each training point.
#     del_Y1 : list of n_var tf.Tensor, each shape (n, 2)
#         Values of gradients of Y1 wrt (x, t) at each training point.

#     Returns
#     -------
#     G : tf.Tensor, shape(n, 1)
#         Value of differential equation at each training point.
#     """
#     # Each of these Tensors is shape (n, 1).
#     n = xt.shape[0]
#     # x = tf.reshape(xt[:, 0], (n, 1))
#     # t = tf.reshape(xt[:, 1], (n, 1))
#     # (rho1, v1x, v1y, v1z, B1y, B1z) = Y1
#     (del_rho1, del_v1x, del_v1y, del_v1z, del_B1y, del_B1z) = del_Y1
#     # drho1_dx = tf.reshape(del_rho1[:, 0], (n, 1))
#     # drho1_dt = tf.reshape(del_rho1[:, 1], (n, 1))
#     # dv1x_dx = tf.reshape(del_v1x[:, 0], (n, 1))
#     # dv1x_dt = tf.reshape(del_v1x[:, 1], (n, 1))
#     # dv1y_dx = tf.reshape(del_v1y[:, 0], (n, 1))
#     # dv1y_dt = tf.reshape(del_v1y[:, 1], (n, 1))
#     dv1z_dx = tf.reshape(del_v1z[:, 0], (n, 1))
#     # dv1z_dt = tf.reshape(del_v1z[:, 1], (n, 1))
#     # dB1y_dx = tf.reshape(del_B1y[:, 0], (n, 1))
#     # dB1y_dt = tf.reshape(del_B1y[:, 1], (n, 1))
#     # dB1z_dx = tf.reshape(del_B1z[:, 0], (n, 1))
#     dB1z_dt = tf.reshape(del_B1z[:, 1], (n, 1))

#     # G is a Tensor of shape (n, 1).
#     G = dB1z_dt - B0x*dv1z_dx
#     return G


# # Make a list of all of the differential equations.
# differential_equations = [
#     pde_rho1,
#     pde_v1x,
#     pde_v1y,
#     pde_v1z,
#     pde_B1y,
#     pde_B1z,
# ]


if __name__ == "__main__":
    print("𝑒 = %s" % 𝑒)
    print("𝑘b = %s" % 𝑘b)
    print("ε0 = %s" % ε0)
    print("𝑚e = %s" % 𝑚e)

    print("independent_variable_names = %s" % independent_variable_names)
    print("independent_variable_labels = %s" % independent_variable_labels)
    print("n_dim = %s" % n_dim)
    print("dependent_variable_names = %s" % dependent_variable_names)
    print("dependent_variable_labels = %s" % dependent_variable_labels)
    print("n_var = %s" % n_var)

    print("%s <= t <= %s" % (t0, t1))
    print("%s <= x <= %s" % (x0, x1))
    print("domain = %s" % domain)

    print("T = %s" % T)
    print("ɣ = %s" % ɣ)

    print("ρ0 = %s" % ρ0)
    print("P0 = %s" % P0)
    print("ux0 = %s" % ux0)
    print("uy0 = %s" % uy0)
    print("uz0 = %s" % uz0)
    print("Bx0 = %s" % Bx0)
    print("By0 = %s" % By0)
    print("Bz0 = %s" % Bz0)

    print("vth = %s" % vth)
    print("ωp = %s" % ωp)
    print("C_Alfven = %s" % C_Alfven)

    print("ρ1_amp = %s" % ρ1_amp)
    print("P1_amp = %s" % P1_amp)
    print("ux1_amp = %s" % ux1_amp)
    print("uy1_amp = %s" % uy1_amp)
    print("uz1_amp = %s" % uz1_amp)
    print("Bx1_amp = %s" % Bx1_amp)
    print("By1_amp = %s" % By1_amp)
    print("Bz1_amp = %s" % Bz1_amp)

    for i in range(nc):
        print("λ[%d] = %s, kx[%d] = %s, ω[%d] = %s, vphase[%d] = %s, f[%d] = %s" %
              (i, λ[i], i, kx[i], i, ω[i], i, vphase[i], i, f[i]))

    nt = 3
    nx = 4
    Xg, Xg_in, Xg_bc = create_training_data_gridded(nt, nx)
    print("Xg = %s" % Xg)
    print("Xg_in = %s" % Xg_in)
    print("Xg_bc = %s" % Xg_bc)

    # bc = compute_boundary_conditions(Xg_bc)
    # print("bc = %s" % bc)
