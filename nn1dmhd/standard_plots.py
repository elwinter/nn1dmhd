"""Routines for standard plots.

This module provides functions which create standard plots.

Author
------
Eric Winter (eric.winter62@gmail.com)

"""


# Import standard modules.

# Import supplemental modules.
import matplotlib.pyplot as plt
import numpy as np

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
