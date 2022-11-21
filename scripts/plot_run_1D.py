#!/usr/bin/env python

# Visualize the results from a pde1bvp_coupled_pinn_1D.py run.


# Import standard modules.
from importlib import import_module
import math as m
import os
import sys

# Import supplemental modules.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
# t_train = tx_train[:, 0]
# x_train = tx_train[:, 1]

# Extract the grid coordinates, and count them.
t_grid = tx_train[::hp.nx_train, 0]  # np.ndarray, shape (p.nt_train,)
x_grid = tx_train[:hp.nx_train, 1]   # np.ndarray, shape (p.nx_train,)

# Load the model-predicted values.
ψ = []
delψ = []
for var_name in p.dependent_variable_names:
    ψ.append(np.loadtxt(os.path.join(problem, "%s_train.dat" % var_name)))
    delψ.append(np.loadtxt(os.path.join(problem, "del_%s_train.dat" % var_name)))

# Load the loss function histories.
losses_model_all = np.loadtxt(os.path.join(problem, "losses_model_all.dat"))
losses_model_bc = np.loadtxt(os.path.join(problem, "losses_model_bc.dat"))
losses_model = np.loadtxt(os.path.join(problem, "losses_model.dat"))
losses_all = np.loadtxt(os.path.join(problem, "losses_all.dat"))
losses_bc = np.loadtxt(os.path.join(problem, "losses_bc.dat"))
losses = np.loadtxt(os.path.join(problem, "losses.dat"))

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

# Compute the figure size for model loss plots, assuming 5x5 for each figure,
# in rows of 2 plots.
loss_figsize = (10, 5*n_rows)

# Compute the figure size for heat maps, assuming 5x5 for each figure,
# in rows of 2 plots.
heatmap_figsize = (10, 5*n_rows)

# Compute the figure size for the start-and-end comparison plots, assuming
# 5x5 for each figure, in rows of 2 plots.
start_end_figsize = (10, 5*p.n_var)

# Create plots in-memory.
mpl.use("AGG")

# Plot the loss history for each model.
plt.figure(figsize=(loss_figsize))
for i in range(p.n_var):
    plt.subplot(n_rows, 2, i + 1)
    variable_name = p.dependent_variable_names[i]
    variable_label = p.dependent_variable_labels[i]
    plt.semilogy(losses_model_all[:, i], label="$L_{all,%s}$" % variable_name)
    plt.semilogy(losses_model_bc[:, i], label="$L_{bc,%s}$" % variable_name)
    plt.semilogy(losses_model[:, i], label="$L_{%s}$" % variable_name)
    plt.title(variable_label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss function")
    plt.legend()
plt.suptitle("Loss function histories by model")
plt.savefig("model_loss.png")
plt.close()

# Plot the total loss function history.
plt.figure()
plt.semilogy(losses_all, label="$L_{all}$")
plt.semilogy(losses_bc, label="$L_{bc}$")
plt.semilogy(losses, label="$L$")
plt.xlabel("Epoch")
plt.ylabel("Loss function")
plt.legend()
plt.grid()
plt.title("Loss function evolution for %s" % problem)
plt.savefig("loss.png")
plt.close()

# Compute the heat map tick locations and labels.
# x will be the horizontal coordinate.
# t will be the vertical coordinate.
n_x_ticks = 5
n_t_ticks = 5
x_tick_pos = np.linspace(0, hp.nx_train - 1, n_x_ticks)
x_tick_labels = ["%.2f" % (p.x0 + x/(hp.nx_train - 1)*(p.x1 - p.x0)) for x in x_tick_pos]
t_tick_pos = np.linspace(0, hp.nt_train - 1, n_t_ticks)
t_tick_labels = list(reversed(["%.2f" % (p.t0 + t/(hp.nt_train - 1)*(p.t1 - p.t0)) for t in t_tick_pos]))

# Plot the analytical solutions.
plt.figure(figsize=heatmap_figsize)
for i in range(p.n_var):
    plt.subplot(n_rows, 2, i + 1)
    variable_name = p.dependent_variable_names[i]
    variable_label = p.dependent_variable_labels[i]
    # For a Seaborn heat map, reshape as (n_t, n_x), then flip.
    Z = np.flip(ψ_a[i].reshape(hp.nt_train, hp.nx_train), axis=0)
    ax = sns.heatmap(Z)
    plt.xticks(x_tick_pos, x_tick_labels)
    plt.yticks(t_tick_pos, t_tick_labels)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(variable_label)
plt.suptitle("Analytical")
plt.savefig("analytical.png")
plt.close()

# Plot the model-predicted solutions.
plt.figure(figsize=heatmap_figsize)
for i in range(p.n_var):
    plt.subplot(n_rows, 2, i + 1)
    variable_name = p.dependent_variable_names[i]
    variable_label = p.dependent_variable_labels[i]
    # For a Seaborn heat map, reshape as (n_t, n_x), then flip.
    Z = np.flip(ψ[i].reshape(hp.nt_train, hp.nx_train), axis=0)
    ax = sns.heatmap(Z)
    plt.xticks(x_tick_pos, x_tick_labels)
    plt.yticks(t_tick_pos, t_tick_labels)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(variable_label)
plt.suptitle("Predicted")
plt.savefig("predicted.png")
plt.close()

# Plot the error in the predicted solutions.
plt.figure(figsize=heatmap_figsize)
for i in range(p.n_var):
    plt.subplot(n_rows, 2, i + 1)
    variable_name = p.dependent_variable_names[0]
    variable_label = p.dependent_variable_labels[i]
    # For a Seaborn heat map, reshape as (n_t, n_x), then flip.
    Z = np.flip(ψ_err[i].reshape(hp.nt_train, hp.nx_train).T, axis=0)
    ax = sns.heatmap(Z)
    plt.xticks(t_tick_pos, t_tick_labels)
    plt.yticks(x_tick_pos, x_tick_labels)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    plt.title("%s, RMS error = %.1e" % (variable_label, rms_err[i]))
plt.suptitle("Error")
plt.savefig("errors.png")
plt.close()

# Compare predicted and analytical solutions at start and end.
plt.figure(figsize=start_end_figsize)
for i in range(p.n_var):
    variable_name = p.dependent_variable_names[i]
    variable_label = p.dependent_variable_labels[i]

    # Compare predicted and analytical values at the starting time.
    plt.subplot(p.n_var, 2, i*2 + 1)
    x = x_grid
    y = ψ[i][:hp.nx_train]
    y_a = ψ_a[i][:hp.nx_train]
    y_err = ψ_err[i][:hp.nx_train]
    rms_err = np.sqrt(np.mean(y_err**2))
    plt.plot(x, y, label="$%s_p$" % variable_name)
    plt.plot(x, y_a, label="$%s_a$" % variable_name)
    plt.plot(x, y_err, label="$%s_{err}$" % variable_name)
    plt.grid()
    plt.legend()
    plt.title("t = %.1f, RMS err = %.1e" % (p.t0, rms_err))
    plt.xlabel(p.independent_variable_labels[0])
    plt.ylabel(variable_label)

    plt.subplot(p.n_var, 2, i*2 + 2)
    y = ψ[i][-hp.nx_train:]
    y_a = ψ_a[i][-hp.nx_train:]
    y_err = ψ_err[i][-hp.nx_train:]
    rms_err = np.sqrt(np.mean(y_err**2))
    plt.plot(x, y, label="$%s_p$" % variable_name)
    plt.plot(x, y_a, label="$%s_a$" % variable_name)
    plt.plot(x, y_err, label="$%s_{err}$" % variable_name)
    plt.grid()
    plt.legend()
    plt.title("t = %.1f, RMS err = %.1e" % (p.t0, rms_err))
    plt.xlabel(p.independent_variable_labels[0])
    plt.ylabel(variable_label)
plt.suptitle("Start and end comparison")
plt.savefig("start_end.png")
plt.close()
