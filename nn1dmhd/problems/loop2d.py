"""Problem definition file for a simple 2-D MHD problem.

This problem definition file describes the 2-D field loop advection problem.
This problem is described at:

https://www.astro.princeton.edu/~jstone/Athena/tests/field-loop/Field-loop.html

NOTE: The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by the TensorFlow
code.

NOTE: In all code, below, the following indices are assigned to physical
independent variables:

    0: x
    1: y
    2: t

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
independent_variable_names = ['x', 'y', 't']

# Labels for independent variables (may use LaTex) - use for plots.
independent_variable_labels = [r'$x$', r'$y$', r'$t$']

# Number of problem dimensions (independent variables).
n_dim = len(independent_variable_names)

# Names of dependent variables.
dependent_variable_names = ['ρ', 'P', 'ux', 'uy', 'uz', 'Bx', 'By', 'Bz']

# Labels for dependent variables (may use LaTex) - use for plots.
dependent_variable_labels = [
    r'$\rho$', r'$P$', r'$u_x$', r'$u_y$', r'$u_z$', r'$B_x$', r'$B_y$', r'$B_z$'
]

# Number of dependent variables.
n_var = len(dependent_variable_names)


# Flow angle clockwise from positive y-axis.
ϴ0 = np.radians(30)

# Define the problem domain.
x0 = -1.0
x1 =  1.0
y0 = -1/(2*np.cos(ϴ0))
y1 =  1/(2*np.cos(ϴ0))
t0 = 0.0
t1 = 1.0
domain = [
    [x0, x1],
    [y0, y1],
    [t0, t1],
]

# Adiabatic index = (N + 2)/N, N = # DOF.
gamma = 5/3

# Normalized physical constants.
μ0 = 1.0  # Permeability of free space

# Magnitude of vector potential.
A = 1e-3

# Initial values for each dependent variable.
ρ0 = 1.0
P0 = 1.0
ux0 = np.sin(ϴ0)
uy0 = np.cos(ϴ0)
uz0 = 0.0
Bx0 = 0.0
By0 = 0.0
Bz0 = 0.0
initial_values = [ρ0, P0, ux0, uy0, uz0, Bx0, By0, Bz0]


def ρ_analytical(xyt:np.ndarray):
    """Compute analytical solution for mass density.

    Compute anaytical solution for mass density.

    Parameters
    ----------
    xyt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    ρ : np.ndarray of float, shape (n,)
        Analytical values for mass density.
    """
    ρ = np.full((xyt.shape[0],), ρ0)
    return ρ


def P_analytical(xyt:np.ndarray):
    """Compute analytical solution for pressure.

    Compute anaytical solution for pressure.

    Parameters
    ----------
    xyt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    P : np.ndarray of float, shape (n,)
        Analytical values for pressure.
    """
    P = np.full((xyt.shape[0],), P0)
    return P


def ux_analytical(xyt:np.ndarray):
    """Compute analytical solution for x-velocity.

    Compute anaytical solution for x-velocity.

    Parameters
    ----------
    xyt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    ux : np.ndarray of float, shape (n,)
        Analytical values for x-velocity.
    """
    ux = np.full((xyt.shape[0],), ux0)
    return ux


def uy_analytical(xyt:np.ndarray):
    """Compute analytical solution for y-velocity.

    Compute anaytical solution for y-velocity.

    Parameters
    ----------
    xyt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    uy : np.ndarray of float, shape (n,)
        Analytical values for y-velocity.
    """
    uy = np.full((xyt.shape[0],), uy0)
    return uy


def uz_analytical(xyt:np.ndarray):
    """Compute analytical solution for z-velocity.

    Compute anaytical solution for z-velocity.

    Parameters
    ----------
    xyt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    uz : np.ndarray of float, shape (n,)
        Analytical values for z-velocity.
    """
    uz = np.full((xyt.shape[0],), uz0)
    return uz


def Bx_analytical(xyt:np.ndarray):
    """Compute analytical solution for x-magnetic field.

    Compute anaytical solution for x-magnetic field.

    Parameters
    ----------
    xyt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    Bx : np.ndarray of float, shape (n,)
        Analytical values for x-magnetic field.
    """
    x = xyt[:, 0]
    y = xyt[:, 1]
    t = xyt[:, 2]
    xc = ux0*t
    yc = uy0*t
    r = np.sqrt((x - xc)**2 + (y - yc)**2)
    Bx = A*y/r
    Bx[np.isclose(r, 0)] = 0
    return Bx


def By_analytical(xyt:np.ndarray):
    """Compute analytical solution for y-magnetic field.

    Compute anaytical solution for y-magnetic field.

    Parameters
    ----------
    xyt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    By : np.ndarray of float, shape (n,)
        Analytical values for y-magnetic field.
    """
    x = xyt[:, 0]
    y = xyt[:, 1]
    t = xyt[:, 2]
    xc = ux0*t
    yc = uy0*t
    r = np.sqrt((x - xc)**2 + (y - yc)**2)
    By = -A*x/r
    By[np.isclose(r, 0)] = 0
    return By


def Bz_analytical(xyt:np.ndarray):
    """Compute analytical solution for z-magnetic field.

    Compute anaytical solution for z-magnetic field.

    Parameters
    ----------
    xyt : np.ndarray of float, shape (n, n_dim)
        Independent variable values for computation.

    Returns
    -------
    Bz : np.ndarray of float, shape (n,)
        Analytical values for z-magnetic field.
    """
    Bz = np.full((xyt.shape[0],), Bz0)
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


def create_training_data_gridded(nx:int, ny:int, nt:int):
    """Create the training data on an evenly-spaced grid.

    Create and return a set of training points evenly spaced in x, y, and
    t. Flatten the data to a list of points. Also return copies of the data
    containing only internal points, and only boundary points.

    Boundary points occur where:
    
    x = x0|x1, y = y0|y1, t = t0|t1

    Parameters
    ----------
    nx, ny, nt : int
        Number of points in x-, y-, and t-dimensions.

    Returns
    -------
    xyt : np.ndarray, shape (nx*ny*nz*nt, n_dim)
        Array of all training points.
    xyt_in : np.ndarray, shape (n_in, n_dim)
        Array of all training points within the boundary.
    xyt_bc : np.ndarray, shape (n_bc, n_dim)
        Array of all training points on the boundary.
    """
    # Create the arrays to hold the training data and the in-domain mask.
    xyt = np.empty((nx, ny, nt, n_dim))
    mask = np.ones((nx, ny, nt), dtype=bool)

    # Compute the individual grid locations along each axis.
    x = np.linspace(x0, x1, nx)
    y = np.linspace(y0, y1, ny)
    t = np.linspace(t0, t1, nt)

    # Compute the coordinates of each training point, and in-domain mask value.
    for (i, xx) in enumerate(x):
        for (j, yy) in enumerate(y):
            for (l, tt) in enumerate(t):
                xyt[i, j, l, :] = (xx, yy, tt)
                if (np.isclose(xx, x0) or np.isclose(xx, x1) or
                    np.isclose(yy, y0) or np.isclose(yy, y1) or
                    np.isclose(tt, t0) or np.isclose(tt, t1)):
                    # This is a boundary point - mask it out.
                    mask[i, j, l] = False

    # Flatten the coordinate and mask lists.
    xyt.shape = (nx*ny*nt, n_dim)
    mask.shape = (nx*ny*nt,)

    # Extract the internal points.
    xyt_in = xyt[mask]

    # Invert the mask and extract the boundary points.
    mask = np.logical_not(mask)
    xyt_bc = xyt[mask]

    # Return the complete, inner, and boundary training points.
    return xyt, xyt_in, xyt_bc


def compute_boundary_conditions(xyt:np.ndarray):
    """Compute the boundary conditions.

    The boundary conditions are computed using the analytical solutions at the
    specified points. Note that this can be used to compute synthetic data
    values within the domain.

    Parameters
    ----------
    xyt : np.ndarray of float, shape (n_bc, n_dim)
        Independent variable values for computation.

    Returns
    -------
    bc : np.ndarray of float, shape (n_bc, n_var)
        Values of each dependent variable at xyt.
    """
    n = len(xyt)
    bc = np.empty((n, n_var))
    for (i, ψa) in enumerate(ψ_analytical):
        bc[:, i] = ψa(xyt)
    return bc


# @tf.function
def pde_ρ(xyt, ψ, del_ψ):
    """Differential equation for mass density.

    Evaluate the differential equation for mass density. This equation is
    derived from the equation of mass continuity.

    Parameters
    ----------
    xyt : tf.Variable, shape (n, n_dim)
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
    n = xyt.shape[0]
    x = tf.reshape(xyt[:, 0], (n, 1))
    y = tf.reshape(xyt[:, 1], (n, 1))
    t = tf.reshape(xyt[:, 2], (n, 1))
    (ρ, P, ux, uy, uz, Bx, By, Bz) = ψ
    (del_ρ, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_ψ
    dρ_dx = tf.reshape(del_ρ[:, 0], (n, 1))
    dρ_dy = tf.reshape(del_ρ[:, 1], (n, 1))
    dρ_dt = tf.reshape(del_ρ[:, 2], (n, 1))
    dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    dP_dy = tf.reshape(del_P[:, 1], (n, 1))
    dP_dt = tf.reshape(del_P[:, 2], (n, 1))
    dux_dx = tf.reshape(del_ux[:, 0], (n, 1))
    dux_dy = tf.reshape(del_ux[:, 1], (n, 1))
    dux_dt = tf.reshape(del_ux[:, 2], (n, 1))
    duy_dx = tf.reshape(del_uy[:, 0], (n, 1))
    duy_dy = tf.reshape(del_uy[:, 1], (n, 1))
    duy_dt = tf.reshape(del_uy[:, 2], (n, 1))
    duz_dx = tf.reshape(del_uz[:, 0], (n, 1))
    duz_dy = tf.reshape(del_uz[:, 1], (n, 1))
    duz_dt = tf.reshape(del_uz[:, 2], (n, 1))
    dBx_dx = tf.reshape(del_Bx[:, 0], (n, 1))
    dBx_dy = tf.reshape(del_Bx[:, 1], (n, 1))
    dBx_dt = tf.reshape(del_Bx[:, 2], (n, 1))
    dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    dBy_dy = tf.reshape(del_By[:, 1], (n, 1))
    dBy_dt = tf.reshape(del_By[:, 2], (n, 1))
    dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    dBz_dy = tf.reshape(del_Bz[:, 1], (n, 1))
    dBz_dt = tf.reshape(del_Bz[:, 2], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        dρ_dt +
        ρ*(dux_dx + duy_dy) +
        dρ_dx*ux + dρ_dy*uy
    )
    return G


# @tf.function
def pde_P(xyt, ψ, del_ψ):
    """Differential equation for P.

    Evaluate the differential equation for pressure (or energy density). This
    equation is derived from the equation of conservation of energy.

    Parameters
    ----------
    xyt : tf.Variable, shape (n, n_dim)
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
    n = xyt.shape[0]
    x = tf.reshape(xyt[:, 0], (n, 1))
    y = tf.reshape(xyt[:, 1], (n, 1))
    t = tf.reshape(xyt[:, 2], (n, 1))
    (ρ, P, ux, uy, uz, Bx, By, Bz) = ψ
    (del_ρ, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_ψ
    dρ_dx = tf.reshape(del_ρ[:, 0], (n, 1))
    dρ_dy = tf.reshape(del_ρ[:, 1], (n, 1))
    dρ_dt = tf.reshape(del_ρ[:, 2], (n, 1))
    dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    dP_dy = tf.reshape(del_P[:, 1], (n, 1))
    dP_dt = tf.reshape(del_P[:, 2], (n, 1))
    dux_dx = tf.reshape(del_ux[:, 0], (n, 1))
    dux_dy = tf.reshape(del_ux[:, 1], (n, 1))
    dux_dt = tf.reshape(del_ux[:, 2], (n, 1))
    duy_dx = tf.reshape(del_uy[:, 0], (n, 1))
    duy_dy = tf.reshape(del_uy[:, 1], (n, 1))
    duy_dt = tf.reshape(del_uy[:, 2], (n, 1))
    duz_dx = tf.reshape(del_uz[:, 0], (n, 1))
    duz_dy = tf.reshape(del_uz[:, 1], (n, 1))
    duz_dt = tf.reshape(del_uz[:, 2], (n, 1))
    dBx_dx = tf.reshape(del_Bx[:, 0], (n, 1))
    dBx_dy = tf.reshape(del_Bx[:, 1], (n, 1))
    dBx_dt = tf.reshape(del_Bx[:, 2], (n, 1))
    dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    dBy_dy = tf.reshape(del_By[:, 1], (n, 1))
    dBy_dt = tf.reshape(del_By[:, 2], (n, 1))
    dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    dBz_dy = tf.reshape(del_Bz[:, 1], (n, 1))
    dBz_dt = tf.reshape(del_Bz[:, 2], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        -gamma*P/ρ*(dρ_dt + ux*dρ_dx + uy*dρ_dy) +
        dP_dt + ux*dP_dx + uy*dP_dy
    )
    return G


# @tf.function
def pde_ux(xyt, ψ, del_ψ):
    """Differential equation for x-velocity.

    Evaluate the differential equation for x-velocity. This equation is derived
    from the equation of conservation of x-momentum.

    Parameters
    ----------
    xyt : tf.Variable, shape (n, n_dim)
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
    n = xyt.shape[0]
    x = tf.reshape(xyt[:, 0], (n, 1))
    y = tf.reshape(xyt[:, 1], (n, 1))
    t = tf.reshape(xyt[:, 2], (n, 1))
    (ρ, P, ux, uy, uz, Bx, By, Bz) = ψ
    (del_ρ, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_ψ
    dρ_dx = tf.reshape(del_ρ[:, 0], (n, 1))
    dρ_dy = tf.reshape(del_ρ[:, 1], (n, 1))
    dρ_dt = tf.reshape(del_ρ[:, 2], (n, 1))
    dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    dP_dy = tf.reshape(del_P[:, 1], (n, 1))
    dP_dt = tf.reshape(del_P[:, 2], (n, 1))
    dux_dx = tf.reshape(del_ux[:, 0], (n, 1))
    dux_dy = tf.reshape(del_ux[:, 1], (n, 1))
    dux_dt = tf.reshape(del_ux[:, 2], (n, 1))
    duy_dx = tf.reshape(del_uy[:, 0], (n, 1))
    duy_dy = tf.reshape(del_uy[:, 1], (n, 1))
    duy_dt = tf.reshape(del_uy[:, 2], (n, 1))
    duz_dx = tf.reshape(del_uz[:, 0], (n, 1))
    duz_dy = tf.reshape(del_uz[:, 1], (n, 1))
    duz_dt = tf.reshape(del_uz[:, 2], (n, 1))
    dBx_dx = tf.reshape(del_Bx[:, 0], (n, 1))
    dBx_dy = tf.reshape(del_Bx[:, 1], (n, 1))
    dBx_dt = tf.reshape(del_Bx[:, 2], (n, 1))
    dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    dBy_dy = tf.reshape(del_By[:, 1], (n, 1))
    dBy_dt = tf.reshape(del_By[:, 2], (n, 1))
    dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    dBz_dy = tf.reshape(del_Bz[:, 1], (n, 1))
    dBz_dt = tf.reshape(del_Bz[:, 2], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        ρ*(dux_dt + ux*dux_dx + uy*dux_dy) +
        dP_dx + (By*(dBy_dx - dBx_dy) + Bz*dBz_dx)/μ0
    )
    return G


# @tf.function
def pde_uy(xyt, ψ, del_ψ):
    """Differential equation for y-velocity.

    Evaluate the differential equation for y-velocity. This equation is derived
    from the equation of conservation of y-momentum.

    Parameters
    ----------
    xyt : tf.Variable, shape (n, n_dim)
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
    n = xyt.shape[0]
    x = tf.reshape(xyt[:, 0], (n, 1))
    y = tf.reshape(xyt[:, 1], (n, 1))
    t = tf.reshape(xyt[:, 2], (n, 1))
    (ρ, P, ux, uy, uz, Bx, By, Bz) = ψ
    (del_ρ, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_ψ
    dρ_dx = tf.reshape(del_ρ[:, 0], (n, 1))
    dρ_dy = tf.reshape(del_ρ[:, 1], (n, 1))
    dρ_dt = tf.reshape(del_ρ[:, 2], (n, 1))
    dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    dP_dy = tf.reshape(del_P[:, 1], (n, 1))
    dP_dt = tf.reshape(del_P[:, 2], (n, 1))
    dux_dx = tf.reshape(del_ux[:, 0], (n, 1))
    dux_dy = tf.reshape(del_ux[:, 1], (n, 1))
    dux_dt = tf.reshape(del_ux[:, 2], (n, 1))
    duy_dx = tf.reshape(del_uy[:, 0], (n, 1))
    duy_dy = tf.reshape(del_uy[:, 1], (n, 1))
    duy_dt = tf.reshape(del_uy[:, 2], (n, 1))
    duz_dx = tf.reshape(del_uz[:, 0], (n, 1))
    duz_dy = tf.reshape(del_uz[:, 1], (n, 1))
    duz_dt = tf.reshape(del_uz[:, 2], (n, 1))
    dBx_dx = tf.reshape(del_Bx[:, 0], (n, 1))
    dBx_dy = tf.reshape(del_Bx[:, 1], (n, 1))
    dBx_dt = tf.reshape(del_Bx[:, 2], (n, 1))
    dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    dBy_dy = tf.reshape(del_By[:, 1], (n, 1))
    dBy_dt = tf.reshape(del_By[:, 2], (n, 1))
    dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    dBz_dy = tf.reshape(del_Bz[:, 1], (n, 1))
    dBz_dt = tf.reshape(del_Bz[:, 2], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        ρ*(duy_dt + ux*duy_dx + uy*duy_dy) +
        dP_dy + (Bz*dBz_dy - Bx*(dBy_dx - dBx_dy))/μ0
    )
    return G


# @tf.function
def pde_uz(xyt, ψ, del_ψ):
    """Differential equation for z-velocity.

    Evaluate the differential equation for z-velocity. This equation is derived
    from the equation of conservation of z-momentum.

    Parameters
    ----------
    xyt : tf.Variable, shape (n, n_dim)
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
    n = xyt.shape[0]
    x = tf.reshape(xyt[:, 0], (n, 1))
    y = tf.reshape(xyt[:, 1], (n, 1))
    t = tf.reshape(xyt[:, 2], (n, 1))
    (ρ, P, ux, uy, uz, Bx, By, Bz) = ψ
    (del_ρ, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_ψ
    dρ_dx = tf.reshape(del_ρ[:, 0], (n, 1))
    dρ_dy = tf.reshape(del_ρ[:, 1], (n, 1))
    dρ_dt = tf.reshape(del_ρ[:, 2], (n, 1))
    dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    dP_dy = tf.reshape(del_P[:, 1], (n, 1))
    dP_dt = tf.reshape(del_P[:, 2], (n, 1))
    dux_dx = tf.reshape(del_ux[:, 0], (n, 1))
    dux_dy = tf.reshape(del_ux[:, 1], (n, 1))
    dux_dt = tf.reshape(del_ux[:, 2], (n, 1))
    duy_dx = tf.reshape(del_uy[:, 0], (n, 1))
    duy_dy = tf.reshape(del_uy[:, 1], (n, 1))
    duy_dt = tf.reshape(del_uy[:, 2], (n, 1))
    duz_dx = tf.reshape(del_uz[:, 0], (n, 1))
    duz_dy = tf.reshape(del_uz[:, 1], (n, 1))
    duz_dt = tf.reshape(del_uz[:, 2], (n, 1))
    dBx_dx = tf.reshape(del_Bx[:, 0], (n, 1))
    dBx_dy = tf.reshape(del_Bx[:, 1], (n, 1))
    dBx_dt = tf.reshape(del_Bx[:, 2], (n, 1))
    dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    dBy_dy = tf.reshape(del_By[:, 1], (n, 1))
    dBy_dt = tf.reshape(del_By[:, 2], (n, 1))
    dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    dBz_dy = tf.reshape(del_Bz[:, 1], (n, 1))
    dBz_dt = tf.reshape(del_Bz[:, 2], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        ρ*(duz_dt + ux*duz_dx + uy*duz_dy) -
        (Bx*dBz_dx + By*dBz_dy)/μ0
    )
    return G


# @tf.function
def pde_Bx(xyt, ψ, del_ψ):
    """Differential equation for x-magnetic field.

    Evaluate the differential equation for x-magnetic field. This equation is
    derived from the x-component of Faraday's Law.

    Parameters
    ----------
    xyt : tf.Variable, shape (n, n_dim)
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
    n = xyt.shape[0]
    x = tf.reshape(xyt[:, 0], (n, 1))
    y = tf.reshape(xyt[:, 1], (n, 1))
    t = tf.reshape(xyt[:, 2], (n, 1))
    (ρ, P, ux, uy, uz, Bx, By, Bz) = ψ
    (del_ρ, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_ψ
    dρ_dx = tf.reshape(del_ρ[:, 0], (n, 1))
    dρ_dy = tf.reshape(del_ρ[:, 1], (n, 1))
    dρ_dt = tf.reshape(del_ρ[:, 2], (n, 1))
    dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    dP_dy = tf.reshape(del_P[:, 1], (n, 1))
    dP_dt = tf.reshape(del_P[:, 2], (n, 1))
    dux_dx = tf.reshape(del_ux[:, 0], (n, 1))
    dux_dy = tf.reshape(del_ux[:, 1], (n, 1))
    dux_dt = tf.reshape(del_ux[:, 2], (n, 1))
    duy_dx = tf.reshape(del_uy[:, 0], (n, 1))
    duy_dy = tf.reshape(del_uy[:, 1], (n, 1))
    duy_dt = tf.reshape(del_uy[:, 2], (n, 1))
    duz_dx = tf.reshape(del_uz[:, 0], (n, 1))
    duz_dy = tf.reshape(del_uz[:, 1], (n, 1))
    duz_dt = tf.reshape(del_uz[:, 2], (n, 1))
    dBx_dx = tf.reshape(del_Bx[:, 0], (n, 1))
    dBx_dy = tf.reshape(del_Bx[:, 1], (n, 1))
    dBx_dt = tf.reshape(del_Bx[:, 2], (n, 1))
    dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    dBy_dy = tf.reshape(del_By[:, 1], (n, 1))
    dBy_dt = tf.reshape(del_By[:, 2], (n, 1))
    dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    dBz_dy = tf.reshape(del_Bz[:, 1], (n, 1))
    dBz_dt = tf.reshape(del_Bz[:, 2], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        dBx_dt + ux*dBx_dx + uy*dBx_dy +
        Bx*duy_dy - By*dux_dy
    )
    return G


# @tf.function
def pde_By(xyt, ψ, del_ψ):
    """Differential equation for y-magnetic field.

    Evaluate the differential equation for y-magnetic field. This equation is
    derived from the y-component of Faraday's Law.

    Parameters
    ----------
    xyt : tf.Variable, shape (n, n_dim)
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
    n = xyt.shape[0]
    x = tf.reshape(xyt[:, 0], (n, 1))
    y = tf.reshape(xyt[:, 1], (n, 1))
    t = tf.reshape(xyt[:, 2], (n, 1))
    (ρ, P, ux, uy, uz, Bx, By, Bz) = ψ
    (del_ρ, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_ψ
    dρ_dx = tf.reshape(del_ρ[:, 0], (n, 1))
    dρ_dy = tf.reshape(del_ρ[:, 1], (n, 1))
    dρ_dt = tf.reshape(del_ρ[:, 2], (n, 1))
    dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    dP_dy = tf.reshape(del_P[:, 1], (n, 1))
    dP_dt = tf.reshape(del_P[:, 2], (n, 1))
    dux_dx = tf.reshape(del_ux[:, 0], (n, 1))
    dux_dy = tf.reshape(del_ux[:, 1], (n, 1))
    dux_dt = tf.reshape(del_ux[:, 2], (n, 1))
    duy_dx = tf.reshape(del_uy[:, 0], (n, 1))
    duy_dy = tf.reshape(del_uy[:, 1], (n, 1))
    duy_dt = tf.reshape(del_uy[:, 2], (n, 1))
    duz_dx = tf.reshape(del_uz[:, 0], (n, 1))
    duz_dy = tf.reshape(del_uz[:, 1], (n, 1))
    duz_dt = tf.reshape(del_uz[:, 2], (n, 1))
    dBx_dx = tf.reshape(del_Bx[:, 0], (n, 1))
    dBx_dy = tf.reshape(del_Bx[:, 1], (n, 1))
    dBx_dt = tf.reshape(del_Bx[:, 2], (n, 1))
    dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    dBy_dy = tf.reshape(del_By[:, 1], (n, 1))
    dBy_dt = tf.reshape(del_By[:, 2], (n, 1))
    dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    dBz_dy = tf.reshape(del_Bz[:, 1], (n, 1))
    dBz_dt = tf.reshape(del_Bz[:, 2], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        dBy_dt + ux*dBy_dx + uy*dBy_dy +
        By*dux_dx - Bx*duy_dx
    )
    return G


# @tf.function
def pde_Bz(xyt, ψ, del_ψ):
    """Differential equation for z-magnetic field.

    Evaluate the differential equation for z-magnetic field. This equation is
    derived from the z-component of Faraday's Law.

    Parameters
    ----------
    xyt : tf.Variable, shape (n, n_dim)
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
    n = xyt.shape[0]
    x = tf.reshape(xyt[:, 0], (n, 1))
    y = tf.reshape(xyt[:, 1], (n, 1))
    t = tf.reshape(xyt[:, 2], (n, 1))
    (ρ, P, ux, uy, uz, Bx, By, Bz) = ψ
    (del_ρ, del_P, del_ux, del_uy, del_uz, del_Bx, del_By, del_Bz) = del_ψ
    dρ_dx = tf.reshape(del_ρ[:, 0], (n, 1))
    dρ_dy = tf.reshape(del_ρ[:, 1], (n, 1))
    dρ_dt = tf.reshape(del_ρ[:, 2], (n, 1))
    dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    dP_dy = tf.reshape(del_P[:, 1], (n, 1))
    dP_dt = tf.reshape(del_P[:, 2], (n, 1))
    dux_dx = tf.reshape(del_ux[:, 0], (n, 1))
    dux_dy = tf.reshape(del_ux[:, 1], (n, 1))
    dux_dt = tf.reshape(del_ux[:, 2], (n, 1))
    duy_dx = tf.reshape(del_uy[:, 0], (n, 1))
    duy_dy = tf.reshape(del_uy[:, 1], (n, 1))
    duy_dt = tf.reshape(del_uy[:, 2], (n, 1))
    duz_dx = tf.reshape(del_uz[:, 0], (n, 1))
    duz_dy = tf.reshape(del_uz[:, 1], (n, 1))
    duz_dt = tf.reshape(del_uz[:, 2], (n, 1))
    dBx_dx = tf.reshape(del_Bx[:, 0], (n, 1))
    dBx_dy = tf.reshape(del_Bx[:, 1], (n, 1))
    dBx_dt = tf.reshape(del_Bx[:, 2], (n, 1))
    dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    dBy_dy = tf.reshape(del_By[:, 1], (n, 1))
    dBy_dt = tf.reshape(del_By[:, 2], (n, 1))
    dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    dBz_dy = tf.reshape(del_Bz[:, 1], (n, 1))
    dBz_dt = tf.reshape(del_Bz[:, 2], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        dBz_dt + ux*dBz_dx + uy*dBz_dy +
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
    nt = 3
    xyt, xyt_in, xyt_bc = create_training_data_gridded(nx, ny, nt)
    print("xyt = %s" % xyt)
    print("xyt_in = %s" % xyt_in)
    print("xyt_bc = %s" % xyt_bc)
    print("Point in domain:")
    for (x, y, t) in (xyt_in):
        print(x, y, t)
    print("Boundary points:")
    for (x, y, t) in (xyt_bc):
        print(x, y, t)

    bc = compute_boundary_conditions(xyt_bc)
    print("bc = %s" % bc)
