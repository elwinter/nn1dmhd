#!/usr/bin/env python

# Create the movie frames for a pde1bvp_coupled_pinn_1D.py run.


# Import standard modules.
from importlib import import_module
import math as m
import os
import sys

# Import supplemental modules.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Specify the run ID (aka problem name).
problem = sys.argv[1]

# Add the subdirectory for the run results to the module search path.
run_path = os.path.join(".", problem)
sys.path.append(run_path)

# Import the problem definition from the run results directory.
p = import_module(problem)

# Read the run hyperparameters from the run results directory.
import hyperparameters as hp

# Load the training points.
tx_train = np.loadtxt(os.path.join(problem, "tx_train.dat"))

# Extract the grid coordinates, and count them.
t_grid = tx_train[::hp.nx_train, 0]  # np.ndarray, shape (p.nt_train,)
x_grid = tx_train[:hp.nx_train, 1]   # np.ndarray, shape (p.nx_train,)

# Load the model-predicted values.
ψ = []
delψ = []
for var_name in p.dependent_variable_names:
    ψ.append(np.loadtxt(os.path.join(problem, "%s_train.dat" % var_name)))
    delψ.append(np.loadtxt(os.path.join(problem, "del_%s_train.dat" % var_name)))

# Compute analytical solutions.
ψ_a = []
for Ya in p.ψ_analytical:
    ψ_a.append(Ya(tx_train))

# Compute the error in the predicted solutions relative to the
# analytical solutions.
ψ_err = [m - a for (m, a) in zip(ψ, ψ_a)]
rms_err = [np.sqrt(np.mean(e**2)) for e in ψ_err]
print("rms_err = %s" % rms_err)

# Compute the number of rows for the 2-per-row plot.
n_rows = m.ceil(p.n_var/2)

# Compute the figure size for the individual frames
frame_figsize = (8, 8)

# Compute the value limits for each variable, considering both the predicted
# and the analytical values.
variable_minimum = [np.amin([Yp, Ya]) for (Yp, Ya) in zip(ψ, ψ_a)]
variable_maximum = [np.amax([Yp, Ya]) for (Yp, Ya) in zip(ψ, ψ_a)]

# Create plots in-memory.
mpl.use("AGG")

# Compare predicted and analytical solutions as a function of time.
for (j, t) in enumerate(t_grid):
    print(j, t)
    j0 = j*hp.nx_train
    j1 = j0 + hp.nx_train
    for (i, variable_name) in enumerate(p.dependent_variable_names):
        print(i, variable_name)
        variable_label = p.dependent_variable_labels[i]

        # Compare predicted and analytical values at the current time.
        x = x_grid
        y = ψ[i][j0:j1]
        y_a = ψ_a[i][j0:j1]
        y_err = ψ_err[i][j0:j1]
        rms_err = np.sqrt(np.mean(y_err**2))
        plt.plot(x, y, label="Predicted")
        plt.plot(x, y_a, label="Analytical")
        plt.plot(x, y_err, label="Error")
        plt.ylim(variable_minimum[i], variable_maximum[i])
        plt.grid()
        plt.legend(loc="upper right")
        plt.title("t = %.2f, RMS err = %.1e" % (t, rms_err))
        plt.xlabel(p.independent_variable_labels[1])
        plt.ylabel(variable_label)
        plt.savefig("%s_%04d.png" % (variable_name, j))
        plt.close()
