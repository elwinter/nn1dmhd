"""Problem definition file for a simple 1-D MHD problem.

This problem definition file describes a static situation: unit pressure and
density, all else is 0. The run should just converge to the initial values for
each quantity.

NOTE: The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by the TensorFlow
code.

NOTE: In all code, below, the following indices are assigned to physical
variables (all are perturbations to steady-state values):

0: rho
1: P
2: vx
3: vy
4: vz
5: Bx
6: By
7: Bz

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard modules.
import numpy as np

# Import supplemental modules.
# import tensorflow as tf

# Import project modules.


# Names of independent variables.
independent_variable_names = ["x", "t"]

# Number of independent variables.
ndim = len(independent_variable_names)

# Names of dependent variables.
dependent_variable_names = ["rho", "P", "vx", "vy", "vz", "Bx", "By", "Bz"]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# Define the problem domain.
x0 = 0.0
x1 = 1.0
t0 = 0.0
t1 = 1.0

# Adiabatic index.
gamma = 3.0

# Initial values for each dependent variable.
rho0 = 1.0
P0 = 1.0
vx0 = 0.0
vy0 = 0.0
vz0 = 0.0
Bx0 = 0.0
By0 = 0.0
Bz0 = 0.0


def rho_analytical(xt:np.ndarray):
    """Compute the analytical solution for mass density.

    Compute the anaytical solution for mass density.

    Parameters
    ----------
    xt : np.ndarray of float, shape (n, ndim)
        Independent variable values for computation.

    Returns
    -------
    rho : np.ndarray of float, shape (n,)
        Analytical values for mass density.
    """
    rho = np.full((xt.shape[0],), rho0)
    return rho


def P_analytical(xt:np.ndarray):
    """Compute the analytical solution for pressure.

    Compute the anaytical solution for pressure.

    Parameters
    ----------
    xt : np.ndarray of float, shape (n, ndim)
        Independent variable values for computation.

    Returns
    -------
    P : np.ndarray of float, shape (n,)
        Analytical values for pressure.
    """
    P = np.full((xt.shape[0],), P0)
    return P


def vx_analytical(xt:np.ndarray):
    """Compute the analytical solution for x-velocity.

    Compute the anaytical solution for x-velocity.

    Parameters
    ----------
    xt : np.ndarray of float, shape (n, ndim)
        Independent variable values for computation.

    Returns
    -------
    vx : np.ndarray of float, shape (n,)
        Analytical values for x-velocity.
    """
    vx = np.full((xt.shape[0],), vx0)
    return vx


def vy_analytical(xt:np.ndarray):
    """Compute the analytical solution for the y-velocity.

    Compute the anaytical solution for the y-velocity.

    Parameters
    ----------
    xt : np.ndarray of float, shape (n, ndim)
        Independent variable values for computation.

    Returns
    -------
    vy : np.ndarray of float, shape (n,)
        Analytical values for y-velocity.
    """
    vy = np.full((xt.shape[0],), vy0)
    return vy


def vz_analytical(xt:np.ndarray):
    """Compute the analytical solution for the z-velocity.

    Compute the anaytical solution for the z-velocity.

    Parameters
    ----------
    xt : np.ndarray of float, shape (n, ndim)
        Independent variable values for computation.

    Returns
    -------
    vz : np.ndarray of float, shape (n,)
        Analytical values for z-velocity.
    """
    vz = np.full((xt.shape[0],), vz0)
    return vz


def Bx_analytical(xt:np.ndarray):
    """Compute the analytical solution for the x-magnetic field.

    Compute the anaytical solution for the x-magnetic field.

    Parameters
    ----------
    xt : np.ndarray of float, shape (n, ndim)
        Independent variable values for computation.

    Returns
    -------
    Bx : np.ndarray of float, shape (n,)
        Analytical values for x-magnetic field.
    """
    Bx = np.full((xt.shape[0],), Bx0)
    return Bx


def By_analytical(xt:np.ndarray):
    """Compute the analytical solution for the y-magnetic field.

    Compute the anaytical solution for the y-magnetic field.

    Parameters
    ----------
    xt : np.ndarray of float, shape (n, ndim)
        Independent variable values for computation.

    Returns
    -------
    By : np.ndarray of float, shape (n,)
        Analytical values for y-magnetic field.
    """
    By = np.full((xt.shape[0],), By0)
    return By


def Bz_analytical(xt:np.ndarray):
    """Compute the analytical solution for the z-magnetic field.

    Compute the anaytical solution for the z-magnetic field.

    Parameters
    ----------
    xt : np.ndarray of float, shape (n, ndim)
        Independent variable values for computation.

    Returns
    -------
    Bz : np.ndarray of float, shape (n,)
        Analytical values for z-magnetic field.
    """
    Bz = np.full((xt.shape[0],), Bz0)
    return Bz


# Gather all of the analytical solutions into a list.
Y_analytical = [
    rho_analytical,
    P_analytical,
    vx_analytical,
    vy_analytical,
    vz_analytical,
    Bx_analytical,
    By_analytical,
    Bz_analytical,
]


def create_training_data(nx:int, nt:int):
    """Create the training data.

    Create and return a set of training points evenly spaced in x and
    t. Flatten the data to a list of pairs of points. Also return copies
    of the data containing only internal points, and only boundary points.

    Boundary conditions are computed for all points where at x = 0 or t = 0.

    Parameters
    ----------
    nx, nt : int
        Number of points in x- and t-dimensions.

    Returns
    -------
    xt : np.ndarray, shape (nx*nt, 2)
        Array of all [x, t] points.
    xt_in : np.ndarray, shape ((nx - 1)*(nt - 1)), 2)
        Array of all [x, t] points within the boundary.
    xt_bc : np.ndarray, shape (nt + nx - 1, 2)
        Array of all [x, t] points on the boundary.
    """
    # Create the array of all training points (x, t), looping over t then x.
    x = np.linspace(x0, x1, nx)
    t = np.linspace(t0, t1, nt)
    X = np.repeat(x, nt)
    T = np.tile(t, nx)
    xt = np.vstack([X, T]).T

    # Now split the training data into two groups - inside the boundary, and
    # on the boundary.

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

    # Return the complete, inner, and boundary training points.
    return xt, xt_in, xt_bc


def compute_boundary_conditions(xt:np.ndarray):
    """Compute the boundary conditions.

    Parameters
    ----------
    xt : np.ndarray of float
        Values of (x, t) on the boundaries, shape (n_bc, 2)

    Returns
    -------
    bc : np.ndarray of float, shape (n_bc, n_var)
        Values of each dependent variable on boundaries.
    """
    n = len(xt)
    bc = np.empty((n, n_var))
    for (i, ya) in enumerate(Y_analytical):
        bc[:, i] = ya(xt)
    return bc


# @tf.function
def pde_rho(xt, Y, del_Y):
    """Differential equation for rho.

    Evaluate the differential equation for rho (density).

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each training point.
    del_Y : list of n_var tf.Tensor, each shape (n, 2)
        Values of gradient wrt (x, t) at each training point, for each
        dependent variable.
    """
    # Each of these Tensors is shape (n, 1).
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    (rho, P, vx, vy, vz, Bx, By, Bz) = Y
    (del_rho, del_P, del_vx, del_vy, del_vz, del_Bx, del_By, del_Bz) = del_Y
    drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
    # dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    dvx_dx = tf.reshape(del_vx[:, 0], (n, 1))
    # dvy_dx = tf.reshape(del_vy[:, 0], (n, 1))
    # dvz_dx = tf.reshape(del_vz[:, 0], (n, 1))
    # dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    # dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
    # dP_dt = tf.reshape(del_P[:, 1], (n, 1))
    # dvx_dt = tf.reshape(del_vx[:, 1], (n, 1))
    # dvy_dt = tf.reshape(del_vy[:, 1], (n, 1))
    # dvz_dt = tf.reshape(del_vz[:, 1], (n, 1))
    # dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 1], (n, 1))
    # G is a Tensor of shape (n, 1).
    G = drho_dt + rho*dvx_dx + drho_dx*vx
    return G

# @tf.function
def pde_P(xt, Y, del_Y):
    """Differential equation for P (actually E).

    Evaluate the differential equation for pressure (or energy density).

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each training point.
    del_Y : list of n_var tf.Tensor, each shape (n, 2)
        Values of gradient wrt (x, t) at each training point, for each
        dependent variable.
    """
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    (rho, P, vx, vy, vz, By, Bz) = Y
    (del_rho, del_P, del_vx, del_vy, del_vz, del_By, del_Bz) = del_Y
    drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
    dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    dvx_dx = tf.reshape(del_vx[:, 0], (n, 1))
    dvy_dx = tf.reshape(del_vy[:, 0], (n, 1))
    dvz_dx = tf.reshape(del_vz[:, 0], (n, 1))
    dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
    dP_dt = tf.reshape(del_P[:, 1], (n, 1))
    dvx_dt = tf.reshape(del_vx[:, 1], (n, 1))
    dvy_dt = tf.reshape(del_vy[:, 1], (n, 1))
    dvz_dt = tf.reshape(del_vz[:, 1], (n, 1))
    dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
    dBz_dt = tf.reshape(del_Bz[:, 1], (n, 1))

    # Compute the total pressure and x-derivative.
    Ptot = P + 0.5*(Bx**2 + By**2 + Bz**2)
    # dBx_dx and dBx_dt are 0.
    dPtot_dx = dP_dx + By*dBy_dx + Bz*dBz_dx
    E = (
        P/(gamma - 1)
        + 0.5*rho*(vx**2 + vy**2 + vz**2)
        + 0.5*(Bx**2 + By**2 + Bz**2)
    )
    dE_dx = (
        dP_dx/(gamma - 1)
        + rho*(vx*dvx_dx + vy*dvy_dx + vz*dvz_dx)
        + drho_dx*0.5*(vx**2 + vy**2  + vz**2)
        + By*dBy_dx + Bz*dBz_dx
    )
    dE_dt = (
        dP_dt/(gamma - 1)
        + rho*(vx*dvx_dt + vy*dvy_dt + vz*dvz_dt)
        + drho_dt*0.5*(vx**2 + vy**2  + vz**2)
        + By*dBy_dt + Bz*dBz_dt
    )
    G = (
        dE_dt + (E + Ptot)*dvx_dx + (dE_dx + dPtot_dx)*vx
        - Bx*(Bx*dvx_dx + By*dvy_dx + dBy_dx*vy + Bz*dvz_dx + dBz_dx*vz)
    )
    return G

# # @tf.function
# def pde_vx(xt, Y, del_Y):
#     """Differential equation for vx.

#     Evaluate the differential equation for vx.

#     Parameters
#     ----------
#     xt : tf.Variable, shape (n, 2)
#         Values of (x, t) at each training point.
#     Y : list of n_var tf.Tensor, each shape (n, 1)
#         Values of dependent variables at each training point.
#     del_Y : list of n_var tf.Tensor, each shape (n, 2)
#         Values of gradient wrt (x, t) at each training point, for each
#         dependent variable.
#     """
#     n = xt.shape[0]
#     # x = tf.reshape(xt[:, 0], (n, 1))
#     # t = tf.reshape(xt[:, 1], (n, 1))
#     (rho, P, vx, vy, vz, By, Bz) = Y
#     (del_rho, del_P, del_vx, del_vy, del_vz, del_By, del_Bz) = del_Y
#     drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
#     dP_dx = tf.reshape(del_P[:, 0], (n, 1))
#     dvx_dx = tf.reshape(del_vx[:, 0], (n, 1))
#     # dvy_dx = tf.reshape(del_vy[:, 0], (n, 1))
#     # dvz_dx = tf.reshape(del_vz[:, 0], (n, 1))
#     dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
#     dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
#     drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
#     # dP_dt = tf.reshape(del_P[:, 1], (n, 1))
#     dvx_dt = tf.reshape(del_vx[:, 1], (n, 1))
#     # dvy_dt = tf.reshape(del_vy[:, 1], (n, 1))
#     # dvz_dt = tf.reshape(del_vz[:, 1], (n, 1))
#     # dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
#     # dBz_dt = tf.reshape(del_Bz[:, 1], (n, 1))
#     # dBx_dx is 0.
#     dPtot_dx = dP_dx + By*dBy_dx + Bz*dBz_dx
#     G = (
#         rho*dvx_dt + drho_dt*vx
#         + rho*2*vx*dvx_dx + drho_dx*vx**2 + dPtot_dx
#     )
#     return G

# # @tf.function
# def pde_vy(xt, Y, del_Y):
#     """Differential equation for vy.

#     Evaluate the differential equation for vy.

#     Parameters
#     ----------
#     xt : tf.Variable, shape (n, 2)
#         Values of (x, t) at each training point.
#     Y : list of n_var tf.Tensor, each shape (n, 1)
#         Values of dependent variables at each training point.
#     del_Y : list of n_var tf.Tensor, each shape (n, 2)
#         Values of gradient wrt (x, t) at each training point, for each
#         dependent variable.
#     """
#     n = xt.shape[0]
#     # x = tf.reshape(xt[:, 0], (n, 1))
#     # t = tf.reshape(xt[:, 1], (n, 1))
#     (rho, P, vx, vy, vz, By, Bz) = Y
#     (del_rho, del_P, del_vx, del_vy, del_vz, del_By, del_Bz) = del_Y
#     drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
#     # dP_dx = tf.reshape(del_P[:, 0], (n, 1))
#     dvx_dx = tf.reshape(del_vx[:, 0], (n, 1))
#     dvy_dx = tf.reshape(del_vy[:, 0], (n, 1))
#     # dvz_dx = tf.reshape(del_vz[:, 0], (n, 1))
#     dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
#     # dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
#     drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
#     # dP_dt = tf.reshape(del_P[:, 1], (n, 1))
#     # dvx_dt = tf.reshape(del_vx[:, 1], (n, 1))
#     dvy_dt = tf.reshape(del_vy[:, 1], (n, 1))
#     # dvz_dt = tf.reshape(del_vz[:, 1], (n, 1))
#     dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
#     # dBz_dt = tf.reshape(del_Bz[:, 1], (n, 1))
#     # dBx_dx is 0.
#     G = (
#         rho*dvy_dt + drho_dt*vy
#         + rho*vx*dvy_dx + rho*dvx_dx*vy + drho_dx*vx*vy
#         - Bx*dBy_dx
#     )
#     return G

# # @tf.function
# def pde_vz(xt, Y, del_Y):
#     """Differential equation for vz.

#     Evaluate the differential equation for vz.

#     Parameters
#     ----------
#     xt : tf.Variable, shape (n, 2)
#         Values of (x, t) at each training point.
#     Y : list of n_var tf.Tensor, each shape (n, 1)
#         Values of dependent variables at each training point.
#     del_Y : list of n_var tf.Tensor, each shape (n, 2)
#         Values of gradient wrt (x, t) at each training point, for each
#         dependent variable.
#     """
#     n = xt.shape[0]
#     # x = tf.reshape(xt[:, 0], (n, 1))
#     # t = tf.reshape(xt[:, 1], (n, 1))
#     (rho, P, vx, vy, vz, By, Bz) = Y
#     (del_rho, del_P, del_vx, del_vy, del_vz, del_By, del_Bz) = del_Y
#     drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
#     # dP_dx = tf.reshape(del_P[:, 0], (n, 1))
#     dvx_dx = tf.reshape(del_vx[:, 0], (n, 1))
#     # dvy_dx = tf.reshape(del_vy[:, 0], (n, 1))
#     dvz_dx = tf.reshape(del_vz[:, 0], (n, 1))
#     # dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
#     dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
#     drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
#     # dP_dt = tf.reshape(del_P[:, 1], (n, 1))
#     dvx_dt = tf.reshape(del_vx[:, 1], (n, 1))
#     # dvy_dt = tf.reshape(del_vy[:, 1], (n, 1))
#     dvz_dt = tf.reshape(del_vz[:, 1], (n, 1))
#     # dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
#     # dBz_dt = tf.reshape(del_Bz[:, 1], (n, 1))
#     # dBx_dx is 0.
#     G = (
#         rho*dvz_dt + drho_dt*vz
#         + rho*vx*dvz_dx + rho*dvx_dx*vz + drho_dx*vx*vz
#         - Bx*dBz_dx
#     )
#     return G

# # @tf.function
# def pde_By(xt, Y, del_Y):
#     """Differential equation for By.

#     Evaluate the differential equation for By.

#     Parameters
#     ----------
#     xt : tf.Variable, shape (n, 2)
#         Values of (x, t) at each training point.
#     Y : list of n_var tf.Tensor, each shape (n, 1)
#         Values of dependent variables at each training point.
#     del_Y : list of n_var tf.Tensor, each shape (n, 2)
#         Values of gradient wrt (x, t) at each training point, for each
#         dependent variable.
#     """
#     n = xt.shape[0]
#     # x = tf.reshape(xt[:, 0], (n, 1))
#     # t = tf.reshape(xt[:, 1], (n, 1))
#     (rho, P, vx, vy, vz, By, Bz) = Y
#     (del_rho, del_P, del_vx, del_vy, del_vz, del_By, del_Bz) = del_Y
#     # drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
#     # dP_dx = tf.reshape(del_P[:, 0], (n, 1))
#     dvx_dx = tf.reshape(del_vx[:, 0], (n, 1))
#     dvy_dx = tf.reshape(del_vy[:, 0], (n, 1))
#     # dvz_dx = tf.reshape(del_vz[:, 0], (n, 1))
#     dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
#     # dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
#     # drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
#     # dP_dt = tf.reshape(del_P[:, 1], (n, 1))
#     # dvx_dt = tf.reshape(del_vx[:, 1], (n, 1))
#     # dvy_dt = tf.reshape(del_vy[:, 1], (n, 1))
#     # dvz_dt = tf.reshape(del_vz[:, 1], (n, 1))
#     dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
#     # dBz_dt = tf.reshape(del_Bz[:, 1], (n, 1))
#     # dBx_dx is 0.
#     G = dBy_dt + By*dvx_dx + dBy_dx*vx - Bx*dvy_dx
#     return G

# # @tf.function
# def pde_Bz(xt, Y, del_Y):
#     """Differential equation for Bz.

#     Evaluate the differential equation for Bz.

#     Parameters
#     ----------
#     xt : tf.Variable, shape (n, 2)
#         Values of (x, t) at each training point.
#     Y : list of n_var tf.Tensor, each shape (n, 1)
#         Values of dependent variables at each training point.
#     del_Y : list of n_var tf.Tensor, each shape (n, 2)
#         Values of gradient wrt (x, t) at each training point, for each
#         dependent variable.
#     """
#     n = xt.shape[0]
#     # x = tf.reshape(xt[:, 0], (n, 1))
#     # t = tf.reshape(xt[:, 1], (n, 1))
#     (rho, P, vx, vy, vz, By, Bz) = Y
#     (del_rho, del_P, del_vx, del_vy, del_vz, del_By, del_Bz) = del_Y
#     # drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
#     # dP_dx = tf.reshape(del_P[:, 0], (n, 1))
#     dvx_dx = tf.reshape(del_vx[:, 0], (n, 1))
#     # dvy_dx = tf.reshape(del_vy[:, 0], (n, 1))
#     # dvz_dx = tf.reshape(del_vz[:, 0], (n, 1))
#     # dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
#     dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
#     # drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
#     # dP_dt = tf.reshape(del_P[:, 1], (n, 1))
#     # dvx_dt = tf.reshape(del_vx[:, 1], (n, 1))
#     # dvy_dt = tf.reshape(del_vy[:, 1], (n, 1))
#     # dvz_dt = tf.reshape(del_vz[:, 1], (n, 1))
#     # dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
#     # dBz_dt = tf.reshape(del_Bz[:, 1], (n, 1))
#     dvx_dx  = del_vx[:, 0]
#     dvz_dx  = del_vz[:, 0]
#     dBz_dx  = del_Bz[:, 0]
#     dBz_dt  = del_Bz[:, 1]
#     G = dBz_dt + Bz*dvx_dx + dBz_dx*vx - Bx*dvz_dx
#     return G


# # Make a list of all of the differential equations.
# differential_equations = [
#     pde_rho,
#     pde_P,
#     pde_vx,
#     pde_vy,
#     pde_vz,
#     pde_By,
#     pde_Bz
# ]


if __name__ == "__main__":
    print("independent_variable_names = %s" % independent_variable_names)
    print("ndim = %s" % ndim)
    print("dependent_variable_names = %s" % dependent_variable_names)
    print("n_var = %s" % n_var)

    print("%s <= x <= %s" % (x0, x1))
    print("%s <= t <= %s" % (t0, t1))

    print("rho0 = %s" % rho0)
    print("P0 = %s" % P0)
    print("vx0 = %s" % vx0)
    print("vy0 = %s" % vy0)
    print("vz0 = %s" % vz0)
    print("Bx0 = %s" % Bx0)
    print("By0 = %s" % By0)
    print("Bz0 = %s" % Bz0)

    nx = nt = 11
    n_in = (nx - 1)*(nt - 1)
    n_bc = nx + nt - 1
    xt, xt_in, xt_bc = create_training_data(nx, nt)
    print("xt = %s" % xt)
    print("xt_in = %s" % xt_in)
    print("xt_bc = %s" % xt_bc)
    assert(len(xt) == nx*nt)
    assert(len(xt_in) == n_in)
    assert(len(xt_bc) == n_bc)

    bc = compute_boundary_conditions(xt_bc)
    print("bc = %s" % bc)
    assert(bc.shape == (n_bc, n_var))
