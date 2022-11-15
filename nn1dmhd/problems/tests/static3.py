"""Problem definition file for a simple 3-D MHD problem.

This problem definition file describes a static situation: unit pressure and
density, all else is 0. The run should just converge to the initial values for
each quantity.

NOTE: The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by the TensorFlow
code.

NOTE: In all code, below, the following indices are assigned to physical
independent variables:

    0: x
    1: y
    2: z
    3: t

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
import numpy as np

# Import supplemental modules.
import tensorflow as tf

# Import project modules.


# Names of independent variables.
independent_variable_names = ['x', 'y', 'z', 't']

# Labels for independent variables (may use LaTex) - use for plots.
independent_variable_labels = [r'$x$', r'$y$', r'$z$', r'$t$']

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
x0 = 0.0
x1 = 1.0
y0 = 0.0
y1 = 1.0
z0 = 0.0
z1 = 1.0
t0 = 0.0
t1 = 1.0
domain = np.array(
    [[x0, x1],
     [y0, y1],
     [z0, z1],
     [t0, t1]]
)

# Adiabatic index = (N + 2)/N, N = # DOF.
gamma = 5/3

# Normalized physical constants.
μ0 = 1.0  # Permeability of free space

# Initial values for each dependent variable.
ρ0 = 1.0
P0 = 1.0
ux0 = 0.0
uy0 = 0.0
uz0 = 0.0
Bx0 = 0.0
By0 = 0.0
Bz0 = 0.0
initial_values = [ρ0, P0, ux0, uy0, uz0, Bx0, By0, Bz0]


def ρ_analytical(xyzt: np.ndarray):
    """Compute analytical solution for mass density.

    Compute anaytical solution for mass density.

    Parameters
    ----------
    xyzt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    ρ : np.ndarray of float, shape (n,)
        Analytical values for mass density.
    """
    ρ = np.full((xyzt.shape[0],), ρ0)
    return ρ


def P_analytical(xyzt: np.ndarray):
    """Compute analytical solution for pressure.

    Compute anaytical solution for pressure.

    Parameters
    ----------
    xyzt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    P : np.ndarray of float, shape (n,)
        Analytical values for pressure.
    """
    P = np.full((xyzt.shape[0],), P0)
    return P


def ux_analytical(xyzt: np.ndarray):
    """Compute analytical solution for x-velocity.

    Compute anaytical solution for x-velocity.

    Parameters
    ----------
    xyzt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    ux : np.ndarray of float, shape (n,)
        Analytical values for x-velocity.
    """
    ux = np.full((xyzt.shape[0],), ux0)
    return ux


def uy_analytical(xyzt: np.ndarray):
    """Compute analytical solution for y-velocity.

    Compute anaytical solution for y-velocity.

    Parameters
    ----------
    xyzt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    uy : np.ndarray of float, shape (n,)
        Analytical values for y-velocity.
    """
    uy = np.full((xyzt.shape[0],), uy0)
    return uy


def uz_analytical(xyzt: np.ndarray):
    """Compute analytical solution for z-velocity.

    Compute anaytical solution for z-velocity.

    Parameters
    ----------
    xyzt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    uz : np.ndarray of float, shape (n,)
        Analytical values for z-velocity.
    """
    uz = np.full((xyzt.shape[0],), uz0)
    return uz


def Bx_analytical(xyzt: np.ndarray):
    """Compute analytical solution for x-magnetic field.

    Compute anaytical solution for x-magnetic field.

    Parameters
    ----------
    xyzt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    Bx : np.ndarray of float, shape (n,)
        Analytical values for x-magnetic field.
    """
    Bx = np.full((xyzt.shape[0],), Bx0)
    return Bx


def By_analytical(xyzt: np.ndarray):
    """Compute analytical solution for y-magnetic field.

    Compute anaytical solution for y-magnetic field.

    Parameters
    ----------
    xyzt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    By : np.ndarray of float, shape (n,)
        Analytical values for y-magnetic field.
    """
    By = np.full((xyzt.shape[0],), By0)
    return By


def Bz_analytical(xyzt: np.ndarray):
    """Compute analytical solution for z-magnetic field.

    Compute anaytical solution for z-magnetic field.

    Parameters
    ----------
    xyzt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    Bz : np.ndarray of float, shape (n,)
        Analytical values for z-magnetic field.
    """
    Bz = np.full((xyzt.shape[0],), Bz0)
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


def create_training_data_gridded(nx: int, ny: int, nz: int, nt: int):
    """Create the training data on an evenly-spaced grid.

    Create and return a set of training points evenly spaced in x, y, z and
    t. Flatten the data to a list of points. Also return copies of the data
    containing only internal points, and only boundary points.

    Boundary points occur where:

    x = x0|x1, y = y0|y1, z = z0|z1, t = t0|t1

    Parameters
    ----------
    nx, ny, nz, nt : int
        Number of points in x-, y-, z-, and t-dimensions.

    Returns
    -------
    xyzt : np.ndarray, shape (nx*ny*nz*nt, n_dim)
        Array of all training points.
    xyzt_in : np.ndarray, shape (n_in, n_dim)
        Array of all training points within the boundary.
    xyzt_bc : np.ndarray, shape (n_bc, n_dim)
        Array of all training points on the boundary.
    """
    # Create the arrays to hold the training data and the in-domain mask.
    xyzt = np.empty((nx, ny, nz, nt, n_dim))
    mask = np.ones((nx, ny, nz, nt), dtype=bool)

    # Compute the individual grid locations along each axis.
    x = np.linspace(x0, x1, nx)
    y = np.linspace(y0, y1, ny)
    z = np.linspace(z0, z1, nz)
    t = np.linspace(t0, t1, nt)

    # Compute the coordinates of each training point, and in-domain mask value.
    for (i, xx) in enumerate(x):
        for (j, yy) in enumerate(y):
            for (k, zz) in enumerate(z):
                for (l, tt) in enumerate(t):
                    xyzt[i, j, k, l, :] = (xx, yy, zz, tt)
                    if (np.isclose(xx, x0) or np.isclose(xx, x1) or
                        np.isclose(yy, y0) or np.isclose(yy, y1) or
                        np.isclose(zz, z0) or np.isclose(zz, z1) or
                        np.isclose(tt, t0) or np.isclose(tt, t1)):
                        # This is a boundary point - mask it out.
                        mask[i, j, k, l] = False

    # Flatten the coordinate and mask lists.
    xyzt.shape = (nx*ny*nz*nt, n_dim)
    mask.shape = (nx*ny*nz*nt,)

    # Extract the internal points.
    xyzt_in = xyzt[mask]

    # Invert the mask and extract the boundary points.
    mask = np.logical_not(mask)
    xyzt_bc = xyzt[mask]

    # Return the complete, inner, and boundary training points.
    return xyzt, xyzt_in, xyzt_bc


def compute_boundary_conditions(xyzt: np.ndarray):
    """Compute the boundary conditions.

    The boundary conditions are computed using the analytical solutions at the
    specified points. Note that this can be used to compute synthetic data
    values within the domain.

    Parameters
    ----------
    xyzt : np.ndarray of float, shape (n_bc, n_dim)
        Independent variable values for computation.

    Returns
    -------
    bc : np.ndarray of float, shape (n_bc, n_var)
        Values of each dependent variable at xyzt.
    """
    n = len(xyzt)
    bc = np.empty((n, n_var))
    for (i, ψa) in enumerate(ψ_analytical):
        bc[:, i] = ψa(xyzt)
    return bc


# @tf.function
def pde_ρ(xyzt, ψ, del_ψ):
    """Differential equation for mass density.

    Evaluate the differential equation for mass density. This equation is
    derived from the equation of mass continuity.

    Parameters
    ----------
    xyzt : tf.Variable, shape (n, n_dim)
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
    n = xyzt.shape[0]
    # x = tf.reshape(xyzt[:, 0], (n, 1))
    # y = tf.reshape(xyzt[:, 1], (n, 1))
    # z = tf.reshape(xyzt[:, 2], (n, 1))
    # t = tf.reshape(xyzt[:, 3], (n, 1))
    (ρ, P, ux, uy, uz, Bx, By, Bz) = ψ
    (del_ρ, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_ψ
    dρ_dx = tf.reshape(del_ρ[:, 0], (n, 1))
    dρ_dy = tf.reshape(del_ρ[:, 1], (n, 1))
    dρ_dz = tf.reshape(del_ρ[:, 2], (n, 1))
    dρ_dt = tf.reshape(del_ρ[:, 3], (n, 1))
    # dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    # dP_dy = tf.reshape(del_P[:, 1], (n, 1))
    # dP_dz = tf.reshape(del_P[:, 2], (n, 1))
    # dP_dt = tf.reshape(del_P[:, 3], (n, 1))
    dux_dx = tf.reshape(del_ux[:, 0], (n, 1))
    # dux_dy = tf.reshape(del_ux[:, 1], (n, 1))
    # dux_dz = tf.reshape(del_ux[:, 2], (n, 1))
    # dux_dt = tf.reshape(del_ux[:, 3], (n, 1))
    # duy_dx = tf.reshape(del_uy[:, 0], (n, 1))
    duy_dy = tf.reshape(del_uy[:, 1], (n, 1))
    # duy_dz = tf.reshape(del_uy[:, 2], (n, 1))
    # duy_dt = tf.reshape(del_uy[:, 3], (n, 1))
    # duz_dx = tf.reshape(del_uz[:, 0], (n, 1))
    # duz_dy = tf.reshape(del_uz[:, 1], (n, 1))
    duz_dz = tf.reshape(del_uz[:, 2], (n, 1))
    # duz_dt = tf.reshape(del_uz[:, 3], (n, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 0], (n, 1))
    # dBx_dy = tf.reshape(del_Bx[:, 1], (n, 1))
    # dBx_dz = tf.reshape(del_Bx[:, 2], (n, 1))
    # dBx_dt = tf.reshape(del_Bx[:, 3], (n, 1))
    # dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    # dBy_dy = tf.reshape(del_By[:, 1], (n, 1))
    # dBy_dz = tf.reshape(del_By[:, 2], (n, 1))
    # dBy_dt = tf.reshape(del_By[:, 3], (n, 1))
    # dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    # dBz_dy = tf.reshape(del_Bz[:, 1], (n, 1))
    # dBz_dz = tf.reshape(del_Bz[:, 2], (n, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 3], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = dρ_dt + ρ*(dux_dx + duy_dy + duz_dz) + dρ_dx*ux + dρ_dy*uy + dρ_dz*uz
    return G


# @tf.function
def pde_P(xyzt, ψ, del_ψ):
    """Differential equation for P.

    Evaluate the differential equation for pressure (or energy density). This
    equation is derived from the equation of conservation of energy.

    Parameters
    ----------
    xyzt : tf.Variable, shape (n, n_dim)
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
    n = xyzt.shape[0]
    # x = tf.reshape(xyzt[:, 0], (n, 1))
    # y = tf.reshape(xyzt[:, 1], (n, 1))
    # z = tf.reshape(xyzt[:, 2], (n, 1))
    # t = tf.reshape(xyzt[:, 3], (n, 1))
    (ρ, P, ux, uy, uz, Bx, By, Bz) = ψ
    (del_ρ, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_ψ
    dρ_dx = tf.reshape(del_ρ[:, 0], (n, 1))
    dρ_dy = tf.reshape(del_ρ[:, 1], (n, 1))
    dρ_dz = tf.reshape(del_ρ[:, 2], (n, 1))
    dρ_dt = tf.reshape(del_ρ[:, 3], (n, 1))
    dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    dP_dy = tf.reshape(del_P[:, 1], (n, 1))
    dP_dz = tf.reshape(del_P[:, 2], (n, 1))
    dP_dt = tf.reshape(del_P[:, 3], (n, 1))
    # dux_dx = tf.reshape(del_ux[:, 0], (n, 1))
    # dux_dy = tf.reshape(del_ux[:, 1], (n, 1))
    # dux_dz = tf.reshape(del_ux[:, 2], (n, 1))
    # dux_dt = tf.reshape(del_ux[:, 3], (n, 1))
    # duy_dx = tf.reshape(del_uy[:, 0], (n, 1))
    # duy_dy = tf.reshape(del_uy[:, 1], (n, 1))
    # duy_dz = tf.reshape(del_uy[:, 2], (n, 1))
    # duy_dt = tf.reshape(del_uy[:, 3], (n, 1))
    # duz_dx = tf.reshape(del_uz[:, 0], (n, 1))
    # duz_dy = tf.reshape(del_uz[:, 1], (n, 1))
    # duz_dz = tf.reshape(del_uz[:, 2], (n, 1))
    # duz_dt = tf.reshape(del_uz[:, 3], (n, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 0], (n, 1))
    # dBx_dy = tf.reshape(del_Bx[:, 1], (n, 1))
    # dBx_dz = tf.reshape(del_Bx[:, 2], (n, 1))
    # dBx_dt = tf.reshape(del_Bx[:, 3], (n, 1))
    # dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    # dBy_dy = tf.reshape(del_By[:, 1], (n, 1))
    # dBy_dz = tf.reshape(del_By[:, 2], (n, 1))
    # dBy_dt = tf.reshape(del_By[:, 3], (n, 1))
    # dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    # dBz_dy = tf.reshape(del_Bz[:, 1], (n, 1))
    # dBz_dz = tf.reshape(del_Bz[:, 2], (n, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 3], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        -gamma*P/ρ*(dρ_dt + ux*dρ_dx + uy*dρ_dy + uz*dρ_dz) + dP_dt +
        ux*dP_dx + uy*dP_dy + uz*dP_dz
    )
    return G


# @tf.function
def pde_ux(xyzt, ψ, del_ψ):
    """Differential equation for x-velocity.

    Evaluate the differential equation for x-velocity. This equation is derived
    from the equation of conservation of x-momentum.

    Parameters
    ----------
    xyzt : tf.Variable, shape (n, n_dim)
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
    n = xyzt.shape[0]
    # x = tf.reshape(xyzt[:, 0], (n, 1))
    # y = tf.reshape(xyzt[:, 1], (n, 1))
    # z = tf.reshape(xyzt[:, 2], (n, 1))
    # t = tf.reshape(xyzt[:, 3], (n, 1))
    (ρ, P, ux, uy, uz, Bx, By, Bz) = ψ
    (del_ρ, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_ψ
    # dρ_dx = tf.reshape(del_ρ[:, 0], (n, 1))
    # dρ_dy = tf.reshape(del_ρ[:, 1], (n, 1))
    # dρ_dz = tf.reshape(del_ρ[:, 2], (n, 1))
    # dρ_dt = tf.reshape(del_ρ[:, 3], (n, 1))
    dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    # dP_dy = tf.reshape(del_P[:, 1], (n, 1))
    # dP_dz = tf.reshape(del_P[:, 2], (n, 1))
    # dP_dt = tf.reshape(del_P[:, 3], (n, 1))
    dux_dx = tf.reshape(del_ux[:, 0], (n, 1))
    dux_dy = tf.reshape(del_ux[:, 1], (n, 1))
    dux_dz = tf.reshape(del_ux[:, 2], (n, 1))
    dux_dt = tf.reshape(del_ux[:, 3], (n, 1))
    # duy_dx = tf.reshape(del_uy[:, 0], (n, 1))
    # duy_dy = tf.reshape(del_uy[:, 1], (n, 1))
    # duy_dz = tf.reshape(del_uy[:, 2], (n, 1))
    # duy_dt = tf.reshape(del_uy[:, 3], (n, 1))
    # duz_dx = tf.reshape(del_uz[:, 0], (n, 1))
    # duz_dy = tf.reshape(del_uz[:, 1], (n, 1))
    # duz_dz = tf.reshape(del_uz[:, 2], (n, 1))
    # duz_dt = tf.reshape(del_uz[:, 3], (n, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 0], (n, 1))
    dBx_dy = tf.reshape(del_Bx[:, 1], (n, 1))
    dBx_dz = tf.reshape(del_Bx[:, 2], (n, 1))
    # dBx_dt = tf.reshape(del_Bx[:, 3], (n, 1))
    dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    # dBy_dy = tf.reshape(del_By[:, 1], (n, 1))
    # dBy_dz = tf.reshape(del_By[:, 2], (n, 1))
    # dBy_dt = tf.reshape(del_By[:, 3], (n, 1))
    dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    # dBz_dy = tf.reshape(del_Bz[:, 1], (n, 1))
    # dBz_dz = tf.reshape(del_Bz[:, 2], (n, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 3], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        ρ*(dux_dt + ux*dux_dx + uy*dux_dy + uz*dux_dz) +
        dP_dx + (By*(dBy_dx - dBx_dy) - Bz*(dBx_dz - dBz_dx))/μ0
    )
    return G


# @tf.function
def pde_uy(xyzt, ψ, del_ψ):
    """Differential equation for y-velocity.

    Evaluate the differential equation for y-velocity. This equation is derived
    from the equation of conservation of y-momentum.

    Parameters
    ----------
    xyzt : tf.Variable, shape (n, n_dim)
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
    n = xyzt.shape[0]
    # x = tf.reshape(xyzt[:, 0], (n, 1))
    # y = tf.reshape(xyzt[:, 1], (n, 1))
    # z = tf.reshape(xyzt[:, 2], (n, 1))
    # t = tf.reshape(xyzt[:, 3], (n, 1))
    (ρ, P, ux, uy, uz, Bx, By, Bz) = ψ
    (del_ρ, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_ψ
    # dρ_dx = tf.reshape(del_ρ[:, 0], (n, 1))
    # dρ_dy = tf.reshape(del_ρ[:, 1], (n, 1))
    # dρ_dz = tf.reshape(del_ρ[:, 2], (n, 1))
    # dρ_dt = tf.reshape(del_ρ[:, 3], (n, 1))
    # dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    dP_dy = tf.reshape(del_P[:, 1], (n, 1))
    # dP_dz = tf.reshape(del_P[:, 2], (n, 1))
    # dP_dt = tf.reshape(del_P[:, 3], (n, 1))
    # dux_dx = tf.reshape(del_ux[:, 0], (n, 1))
    # dux_dy = tf.reshape(del_ux[:, 1], (n, 1))
    # dux_dz = tf.reshape(del_ux[:, 2], (n, 1))
    # dux_dt = tf.reshape(del_ux[:, 3], (n, 1))
    duy_dx = tf.reshape(del_uy[:, 0], (n, 1))
    duy_dy = tf.reshape(del_uy[:, 1], (n, 1))
    duy_dz = tf.reshape(del_uy[:, 2], (n, 1))
    duy_dt = tf.reshape(del_uy[:, 3], (n, 1))
    # duz_dx = tf.reshape(del_uz[:, 0], (n, 1))
    # duz_dy = tf.reshape(del_uz[:, 1], (n, 1))
    # duz_dz = tf.reshape(del_uz[:, 2], (n, 1))
    # duz_dt = tf.reshape(del_uz[:, 3], (n, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 0], (n, 1))
    dBx_dy = tf.reshape(del_Bx[:, 1], (n, 1))
    # dBx_dz = tf.reshape(del_Bx[:, 2], (n, 1))
    # dBx_dt = tf.reshape(del_Bx[:, 3], (n, 1))
    dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    # dBy_dy = tf.reshape(del_By[:, 1], (n, 1))
    dBy_dz = tf.reshape(del_By[:, 2], (n, 1))
    # dBy_dt = tf.reshape(del_By[:, 3], (n, 1))
    # dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    dBz_dy = tf.reshape(del_Bz[:, 1], (n, 1))
    # dBz_dz = tf.reshape(del_Bz[:, 2], (n, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 3], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        ρ*(duy_dt + ux*duy_dx + uy*duy_dy + uz*duy_dz) +
        dP_dy + (Bz*(dBz_dy - dBy_dz) - Bx*(dBy_dx - dBx_dy))/μ0
    )
    return G


# @tf.function
def pde_uz(xyzt, ψ, del_ψ):
    """Differential equation for z-velocity.

    Evaluate the differential equation for z-velocity. This equation is derived
    from the equation of conservation of z-momentum.

    Parameters
    ----------
    xyzt : tf.Variable, shape (n, n_dim)
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
    n = xyzt.shape[0]
    # x = tf.reshape(xyzt[:, 0], (n, 1))
    # y = tf.reshape(xyzt[:, 1], (n, 1))
    # z = tf.reshape(xyzt[:, 2], (n, 1))
    # t = tf.reshape(xyzt[:, 3], (n, 1))
    (ρ, P, ux, uy, uz, Bx, By, Bz) = ψ
    (del_ρ, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_ψ
    # dρ_dx = tf.reshape(del_ρ[:, 0], (n, 1))
    # dρ_dy = tf.reshape(del_ρ[:, 1], (n, 1))
    # dρ_dz = tf.reshape(del_ρ[:, 2], (n, 1))
    # dρ_dt = tf.reshape(del_ρ[:, 3], (n, 1))
    # dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    # dP_dy = tf.reshape(del_P[:, 1], (n, 1))
    dP_dz = tf.reshape(del_P[:, 2], (n, 1))
    # dP_dt = tf.reshape(del_P[:, 3], (n, 1))
    # dux_dx = tf.reshape(del_ux[:, 0], (n, 1))
    # dux_dy = tf.reshape(del_ux[:, 1], (n, 1))
    # dux_dz = tf.reshape(del_ux[:, 2], (n, 1))
    # dux_dt = tf.reshape(del_ux[:, 3], (n, 1))
    # duy_dx = tf.reshape(del_uy[:, 0], (n, 1))
    # duy_dy = tf.reshape(del_uy[:, 1], (n, 1))
    # duy_dz = tf.reshape(del_uy[:, 2], (n, 1))
    # duy_dt = tf.reshape(del_uy[:, 3], (n, 1))
    duz_dx = tf.reshape(del_uz[:, 0], (n, 1))
    duz_dy = tf.reshape(del_uz[:, 1], (n, 1))
    duz_dz = tf.reshape(del_uz[:, 2], (n, 1))
    duz_dt = tf.reshape(del_uz[:, 3], (n, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 0], (n, 1))
    # dBx_dy = tf.reshape(del_Bx[:, 1], (n, 1))
    dBx_dz = tf.reshape(del_Bx[:, 2], (n, 1))
    # dBx_dt = tf.reshape(del_Bx[:, 3], (n, 1))
    # dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    # dBy_dy = tf.reshape(del_By[:, 1], (n, 1))
    dBy_dz = tf.reshape(del_By[:, 2], (n, 1))
    # dBy_dt = tf.reshape(del_By[:, 3], (n, 1))
    dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    dBz_dy = tf.reshape(del_Bz[:, 1], (n, 1))
    # dBz_dz = tf.reshape(del_Bz[:, 2], (n, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 3], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        ρ*(duz_dt + ux*duz_dx + uy*duz_dy + uz*duz_dz) +
        dP_dz + (Bx*(dBx_dz - dBz_dx) - By*(dBz_dy - dBy_dz))/μ0
    )
    return G


# @tf.function
def pde_Bx(xyzt, ψ, del_ψ):
    """Differential equation for x-magnetic field.

    Evaluate the differential equation for x-magnetic field. This equation is
    derived from the x-component of Faraday's Law.

    Parameters
    ----------
    xyzt : tf.Variable, shape (n, n_dim)
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
    n = xyzt.shape[0]
    # x = tf.reshape(xyzt[:, 0], (n, 1))
    # y = tf.reshape(xyzt[:, 1], (n, 1))
    # z = tf.reshape(xyzt[:, 2], (n, 1))
    # t = tf.reshape(xyzt[:, 3], (n, 1))
    (ρ, P, ux, uy, uz, Bx, By, Bz) = ψ
    (del_ρ, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_ψ
    # dρ_dx = tf.reshape(del_ρ[:, 0], (n, 1))
    # dρ_dy = tf.reshape(del_ρ[:, 1], (n, 1))
    # dρ_dz = tf.reshape(del_ρ[:, 2], (n, 1))
    # dρ_dt = tf.reshape(del_ρ[:, 3], (n, 1))
    # dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    # dP_dy = tf.reshape(del_P[:, 1], (n, 1))
    # dP_dz = tf.reshape(del_P[:, 2], (n, 1))
    # dP_dt = tf.reshape(del_P[:, 3], (n, 1))
    # dux_dx = tf.reshape(del_ux[:, 0], (n, 1))
    dux_dy = tf.reshape(del_ux[:, 1], (n, 1))
    dux_dz = tf.reshape(del_ux[:, 2], (n, 1))
    # dux_dt = tf.reshape(del_ux[:, 3], (n, 1))
    # duy_dx = tf.reshape(del_uy[:, 0], (n, 1))
    duy_dy = tf.reshape(del_uy[:, 1], (n, 1))
    # duy_dz = tf.reshape(del_uy[:, 2], (n, 1))
    # duy_dt = tf.reshape(del_uy[:, 3], (n, 1))
    # duz_dx = tf.reshape(del_uz[:, 0], (n, 1))
    # duz_dy = tf.reshape(del_uz[:, 1], (n, 1))
    duz_dz = tf.reshape(del_uz[:, 2], (n, 1))
    # duz_dt = tf.reshape(del_uz[:, 3], (n, 1))
    dBx_dx = tf.reshape(del_Bx[:, 0], (n, 1))
    dBx_dy = tf.reshape(del_Bx[:, 1], (n, 1))
    dBx_dz = tf.reshape(del_Bx[:, 2], (n, 1))
    dBx_dt = tf.reshape(del_Bx[:, 3], (n, 1))
    # dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    # dBy_dy = tf.reshape(del_By[:, 1], (n, 1))
    # dBy_dz = tf.reshape(del_By[:, 2], (n, 1))
    # dBy_dt = tf.reshape(del_By[:, 3], (n, 1))
    # dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    # dBz_dy = tf.reshape(del_Bz[:, 1], (n, 1))
    # dBz_dz = tf.reshape(del_Bz[:, 2], (n, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 3], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        dBx_dt + ux*dBx_dx + uy*dBx_dy + uz*dBx_dz +
        Bx*(duy_dy + duz_dz) - By*dux_dy - Bz*dux_dz
    )
    return G


# @tf.function
def pde_By(xyzt, ψ, del_ψ):
    """Differential equation for y-magnetic field.

    Evaluate the differential equation for y-magnetic field. This equation is
    derived from the y-component of Faraday's Law.

    Parameters
    ----------
    xyzt : tf.Variable, shape (n, n_dim)
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
    n = xyzt.shape[0]
    # x = tf.reshape(xyzt[:, 0], (n, 1))
    # y = tf.reshape(xyzt[:, 1], (n, 1))
    # z = tf.reshape(xyzt[:, 2], (n, 1))
    # t = tf.reshape(xyzt[:, 3], (n, 1))
    (ρ, P, ux, uy, uz, Bx, By, Bz) = ψ
    (del_ρ, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_ψ
    # dρ_dx = tf.reshape(del_ρ[:, 0], (n, 1))
    # dρ_dy = tf.reshape(del_ρ[:, 1], (n, 1))
    # dρ_dz = tf.reshape(del_ρ[:, 2], (n, 1))
    # dρ_dt = tf.reshape(del_ρ[:, 3], (n, 1))
    # dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    # dP_dy = tf.reshape(del_P[:, 1], (n, 1))
    # dP_dz = tf.reshape(del_P[:, 2], (n, 1))
    # dP_dt = tf.reshape(del_P[:, 3], (n, 1))
    dux_dx = tf.reshape(del_ux[:, 0], (n, 1))
    # dux_dy = tf.reshape(del_ux[:, 1], (n, 1))
    # dux_dz = tf.reshape(del_ux[:, 2], (n, 1))
    # dux_dt = tf.reshape(del_ux[:, 3], (n, 1))
    duy_dx = tf.reshape(del_uy[:, 0], (n, 1))
    # duy_dy = tf.reshape(del_uy[:, 1], (n, 1))
    duy_dz = tf.reshape(del_uy[:, 2], (n, 1))
    # duy_dt = tf.reshape(del_uy[:, 3], (n, 1))
    # duz_dx = tf.reshape(del_uz[:, 0], (n, 1))
    # duz_dy = tf.reshape(del_uz[:, 1], (n, 1))
    duz_dz = tf.reshape(del_uz[:, 2], (n, 1))
    # duz_dt = tf.reshape(del_uz[:, 3], (n, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 0], (n, 1))
    # dBx_dy = tf.reshape(del_Bx[:, 1], (n, 1))
    # dBx_dz = tf.reshape(del_Bx[:, 2], (n, 1))
    # dBx_dt = tf.reshape(del_Bx[:, 3], (n, 1))
    dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    dBy_dy = tf.reshape(del_By[:, 1], (n, 1))
    dBy_dz = tf.reshape(del_By[:, 2], (n, 1))
    dBy_dt = tf.reshape(del_By[:, 3], (n, 1))
    # dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    # dBz_dy = tf.reshape(del_Bz[:, 1], (n, 1))
    # dBz_dz = tf.reshape(del_Bz[:, 2], (n, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 3], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        dBy_dt + ux*dBy_dx + uy*dBy_dy + uz*dBy_dz +
        By*(dux_dx + duz_dz) - Bx*duy_dx - Bz*duy_dz
    )
    return G


# @tf.function
def pde_Bz(xyzt, ψ, del_ψ):
    """Differential equation for z-magnetic field.

    Evaluate the differential equation for z-magnetic field. This equation is
    derived from the z-component of Faraday's Law.

    Parameters
    ----------
    xyzt : tf.Variable, shape (n, n_dim)
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
    n = xyzt.shape[0]
    # x = tf.reshape(xyzt[:, 0], (n, 1))
    # y = tf.reshape(xyzt[:, 1], (n, 1))
    # z = tf.reshape(xyzt[:, 2], (n, 1))
    # t = tf.reshape(xyzt[:, 3], (n, 1))
    (ρ, P, ux, uy, uz, Bx, By, Bz) = ψ
    (del_ρ, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_ψ
    # dρ_dx = tf.reshape(del_ρ[:, 0], (n, 1))
    # dρ_dy = tf.reshape(del_ρ[:, 1], (n, 1))
    # dρ_dz = tf.reshape(del_ρ[:, 2], (n, 1))
    # dρ_dt = tf.reshape(del_ρ[:, 3], (n, 1))
    # dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    # dP_dy = tf.reshape(del_P[:, 1], (n, 1))
    # dP_dz = tf.reshape(del_P[:, 2], (n, 1))
    # dP_dt = tf.reshape(del_P[:, 3], (n, 1))
    dux_dx = tf.reshape(del_ux[:, 0], (n, 1))
    # dux_dy = tf.reshape(del_ux[:, 1], (n, 1))
    # dux_dz = tf.reshape(del_ux[:, 2], (n, 1))
    # dux_dt = tf.reshape(del_ux[:, 3], (n, 1))
    # duy_dx = tf.reshape(del_uy[:, 0], (n, 1))
    duy_dy = tf.reshape(del_uy[:, 1], (n, 1))
    # duy_dz = tf.reshape(del_uy[:, 2], (n, 1))
    # duy_dt = tf.reshape(del_uy[:, 3], (n, 1))
    duz_dx = tf.reshape(del_uz[:, 0], (n, 1))
    duz_dy = tf.reshape(del_uz[:, 1], (n, 1))
    # duz_dz = tf.reshape(del_uz[:, 2], (n, 1))
    # duz_dt = tf.reshape(del_uz[:, 3], (n, 1))
    # dBx_dx = tf.reshape(del_Bx[:, 0], (n, 1))
    # dBx_dy = tf.reshape(del_Bx[:, 1], (n, 1))
    # dBx_dz = tf.reshape(del_Bx[:, 2], (n, 1))
    # dBx_dt = tf.reshape(del_Bx[:, 3], (n, 1))
    # dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    # dBy_dy = tf.reshape(del_By[:, 1], (n, 1))
    # dBy_dz = tf.reshape(del_By[:, 2], (n, 1))
    # dBy_dt = tf.reshape(del_By[:, 3], (n, 1))
    dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    dBz_dy = tf.reshape(del_Bz[:, 1], (n, 1))
    dBz_dz = tf.reshape(del_Bz[:, 2], (n, 1))
    dBz_dt = tf.reshape(del_Bz[:, 3], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        dBz_dt + ux*dBz_dx + uy*dBz_dy + uz*dBz_dz +
        Bz*(dux_dx + duy_dy) - Bx*duz_dx - By*duz_dy
    )
    return G


# Make a list of all of the differential equations.
differential_equations = [
    pde_ρ,
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

    print("%s <= x <= %s" % (x0, x1))
    print("%s <= y <= %s" % (y0, y1))
    print("%s <= z <= %s" % (z0, z1))
    print("%s <= t <= %s" % (t0, t1))

    print("ρ0 = %s" % ρ0)
    print("P0 = %s" % P0)
    print("ux0 = %s" % ux0)
    print("uy0 = %s" % uy0)
    print("uz0 = %s" % uz0)
    print("Bx0 = %s" % Bx0)
    print("By0 = %s" % By0)
    print("Bz0 = %s" % Bz0)

    nx = 6
    ny = 5
    nz = 4
    nt = 3
    xyzt, xyzt_in, xyzt_bc = create_training_data_gridded(nx, ny, nz, nt)
    print("xyzt = %s" % xyzt)
    print("xyzt_in = %s" % xyzt_in)
    print("xyzt_bc = %s" % xyzt_bc)
    print("Point in domain:")
    for (x, y, z, t) in (xyzt_in):
        print(x, y, z, t)
    print("Boundary points:")
    for (x, y, z, t) in (xyzt_bc):
        print(x, y, z, t)

    bc = compute_boundary_conditions(xyzt_bc)
    print("bc = %s" % bc)
