"""Routines for standard plots.

This module provides functions which create standard plots.

Author
------
Eric Winter (eric.winter62@gmail.com)

"""


# Import standard modules.

# Import supplemental modules.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Import project modules.


def plot_loss_functions(L_res, L_data, L, ax, title="Loss functions"):
    """Plot the standard triplet of PINN loss functions.

    Plot the standard triplet of PINN loss functions.

    Parameters
    ----------
    L_res, L_data, L : np.ndarray, shape (n,)
        Loss functions for residual, data, and total.
    ax : matplotlib.Axes
        Axes object to use for the plot.
    title : str (optional, default "Loss functions")
        Title for plot

    Returns
    -------
    None
    """
    ax.semilogy(L_res, label="$L_{res}$")
    ax.semilogy(L_data, label="$L_{data}$")
    ax.semilogy(L, label="$L$")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss function")
    ax.grid()
    ax.legend()
    ax.set_title(title)


def plot_BxBy_quiver(x, y, Bx, By, ax, title="Magnetic field"):
    """Plot Bx and By components as a quiver plot.

    Plot Bx and By components as a quiver plot.

    Parameters
    ----------
    x, y : np.ndarray, shape (n,)
        x and y coordinates for arrows.
    Bx, By : np.ndarray, shape ( n,)
        x- and y-components for arrows.
    ax : matplotlib.Axes
        Axes object to use for the plot.
    title : str (optional, default "Magnetic field")
        Title for plot

    Returns
    -------
    None
    """
    ax.quiver(x, y, Bx, By)
    ax.set_aspect(1.0)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.grid()
    ax.set_title(title)


def plot_logarithmic_heatmap(z, ax, title="Logarithmic heat map"):
    """Plot a logarithmic heat map.

    Plot a logarithmic heat map.

    Parameters
    ----------
    z : np.ndarray, shape (n,)
        Values for heatmap.
    ax : matplotlib.Axes
        Axes object to use for the plot.
    title : str (optional, default "Logarithmic heat map")
        Title for plot

    Returns
    -------
    None
    """
    sns.heatmap(z, ax=ax, norm=mpl.colors.LogNorm(), square=True)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.grid()
    ax.set_title(title)


def plot_linear_heatmap(z, ax, title="Linear heat map"):
    """Plot a linear heat map.

    Plot a linear heat map.

    Parameters
    ----------
    z : np.ndarray, shape (n,)
        Values for heatmap.
    ax : matplotlib.Axes
        Axes object to use for the plot.
    title : str (optional, default "Logarithmic heat map")
        Title for plot

    Returns
    -------
    None
    """
    sns.heatmap(z, ax=ax, square=True)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.grid()
    ax.set_title(title)
