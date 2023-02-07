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
    L_res, L_data, L : np.ndarray, shape ( n,)
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
    x, y : np.ndarray, shape ( n,)
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
    # ax.set_xticks(XY_x_tick_pos, XY_x_tick_labels)
    ax.set_ylabel("$y$")
    # ax.set_yticks(XY_y_tick_pos, XY_y_tick_labels)
    ax.grid()
    ax.set_title(title)


def plot_logarithmic_heatmap(z, ax, title="Heat map"):
    """Plot Bx and By components as a quiver plot.

    Plot Bx and By components as a quiver plot.

    Parameters
    ----------
    x, y : np.ndarray, shape ( n,)
        x and y coordinates for arrows.
    Bx, By : np.ndarray, shape ( n,)
        x- and y-components for arrows.
    ax : matplotlib.Axes
        Axes object to use for the plot.
    title : str (optional, default "Magnetic field")
        Title for plot

    Returns
    -------
    # None
    """
    sns.heatmap(z, ax=ax, norm=mpl.colors.LogNorm(), square=True)
    ax.set_xlabel("$x$")
    # ax.set_xticks(heatmap_x_tick_pos, heatmap_x_tick_labels, rotation=0)
    ax.set_ylabel("$y$")
    # ax.set_yticks(heatmap_y_tick_pos, heatmap_y_tick_labels)
    ax.grid()
    ax.set_title(title)
