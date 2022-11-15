#!/usr/bin/env python

# Visualize the results from a pde1bvp_coupled_pinn_2D.py run.


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
runid = "static2"

# Add the subdirectory for the run results to the module search path.
run_path = os.path.join(".", runid)
sys.path.append(run_path)

# Import the problem definition from the run results directory.
p = import_module(runid)

# Read the run hyperparameters from the run results directory.
import hyperparameters as hp

# Load the training points.
xyt_train = np.loadtxt(os.path.join(runid, "xyt_train.dat"))
x_train = xyt_train[:, 0]
y_train = xyt_train[:, 1]
t_train = xyt_train[:, 2]

# Extract the unique training point values.
x_train_vals = x_train[::hp.ny_train*hp.nt_train]
y_train_vals = y_train[:hp.nx_train*hp.nt_train:hp.nt_train]
t_train_vals = t_train[:hp.nt_train]
n_x_train_vals = len(x_train_vals)
n_y_train_vals = len(y_train_vals)
n_t_train_vals = len(t_train_vals)

# Load the model-predicted values.
ψ = []
delψ = []
for var_name in p.dependent_variable_names:
    ψ.append(np.loadtxt(os.path.join(runid, "%s_train.dat" % var_name)))
    delψ.append(np.loadtxt(os.path.join(runid, "del_%s_train.dat" % var_name)))

# Load the loss function histories.
losses_model_all = np.loadtxt(os.path.join(runid, "losses_model_all.dat"))
losses_model_bc = np.loadtxt(os.path.join(runid, "losses_model_bc.dat"))
losses_model = np.loadtxt(os.path.join(runid, "losses_model.dat"))
losses_all = np.loadtxt(os.path.join(runid, "losses_all.dat"))
losses_bc = np.loadtxt(os.path.join(runid, "losses_bc.dat"))
losses = np.loadtxt(os.path.join(runid, "losses.dat"))

# Compute analytical solutions.
ψ_a = []
for Ya in p.ψ_analytical:
    ψ_a.append(Ya(xyt_train))

# Compute the error in the predicted solutions relative to the
# analytical solutions.
ψ_err = [m - a for (m, a) in zip(ψ, ψ_a)]
rms_err = [np.sqrt(np.mean(e**2)) for e in ψ_err]

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
plt.title("Loss function evolution for %s" % runid)
plt.savefig("loss.png")
plt.close()

# Compute the heat map tick locations and labels.
n_x_ticks = 5
n_y_ticks = 5
n_t_ticks = 5
x_tick_pos = np.linspace(0, n_x_train_vals - 1, n_x_ticks)
x_tick_labels = ["%.1f" % (x/(n_x_train_vals - 1)) for x in x_tick_pos]
y_tick_pos = np.linspace(0, n_y_train_vals - 1, n_y_ticks)
y_tick_labels = ["%.1f" % (y/(n_y_train_vals - 1)) for y in y_tick_pos]
t_tick_pos = np.linspace(0, n_t_train_vals - 1, n_t_ticks)
t_tick_labels = ["%.1f" % (p.t0 + t/(n_t_train_vals - 1)*(p.t1 - p.t0)) for t in t_tick_pos]
t_tick_labels = list(reversed(t_tick_labels))
print(x_tick_labels, y_tick_labels, t_tick_labels)

# Plot the model-predicted solutions at the last time.
plt.figure(figsize=heatmap_figsize)
for i in range(p.n_var):
    plt.subplot(n_rows, 2, i + 1)
    variable_name = p.dependent_variable_names[i]
    variable_label = p.dependent_variable_labels[i]
    # For a Seaborn heat map, reshape as (n_x, n_y), then transpose, then flip.
    ψlast = ψ[i][::hp.nt_train]
    Z = np.flip(ψlast.reshape(hp.nx_train, hp.ny_train).T, axis=0)
    ax = sns.heatmap(Z)
    plt.xticks(x_tick_pos, x_tick_labels)
    plt.yticks(y_tick_pos, y_tick_labels)
    ax.set_xlabel(p.independent_variable_labels[0])
    ax.set_ylabel(p.independent_variable_labels[1])
    plt.title(variable_label)
plt.suptitle("Predicted at final time")
plt.savefig("models.png")
plt.close()

# Plot the analytical solutions at the last time.
plt.figure(figsize=heatmap_figsize)
for i in range(p.n_var):
    plt.subplot(n_rows, 2, i + 1)
    variable_name = p.dependent_variable_names[i]
    variable_label = p.dependent_variable_labels[i]
    # For a Seaborn heat map, reshape as (n_x, n_y), then transpose, then flip.
    ψlast = ψ_a[i][::hp.nt_train]
    Z = np.flip(ψlast.reshape(hp.nx_train, hp.ny_train).T, axis=0)
    ax = sns.heatmap(Z)
    plt.xticks(x_tick_pos, x_tick_labels)
    plt.yticks(y_tick_pos, y_tick_labels)
    ax.set_xlabel(p.independent_variable_labels[0])
    ax.set_ylabel(p.independent_variable_labels[1])
    plt.title(variable_label)
plt.suptitle("Analytical at final time")
plt.savefig("analytical.png")
plt.close()

# Plot the error in the predicted solutions at the last time.
plt.figure(figsize=heatmap_figsize)
for i in range(p.n_var):
    plt.subplot(n_rows, 2, i + 1)
    variable_name = p.dependent_variable_names[0]
    variable_label = p.dependent_variable_labels[i]
    # For a Seaborn heat map, reshape as (n_x, n_y), then transpose, then flip.
    ψlast = ψ_err[i][::hp.nt_train]
    Z = np.flip(ψlast.reshape(hp.nx_train, hp.ny_train).T, axis=0)
    ax = sns.heatmap(Z)
    plt.xticks(x_tick_pos, x_tick_labels)
    plt.yticks(y_tick_pos, y_tick_labels)
    ax.set_xlabel(p.independent_variable_labels[0])
    ax.set_ylabel(p.independent_variable_labels[1])
    plt.title("%s, RMS error = %.1e" % (variable_label, rms_err[i]))
plt.suptitle("Error at final time")
plt.savefig("errors.png")
plt.close()
