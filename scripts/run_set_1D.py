#!/usr/bin/env python

# Run a set of training sessions.


# Import standard modules.
import os
import sys
import time


# Specify the path to the problem file.
problem = sys.argv[1]

# Specify set parameters.
n_layerss = [1]
Hs = [10]
learning_rates = [0.01]
nt_trains = [10]
nx_trains = [10]
w_bcs = [0.95]
max_epochss = [100]

# Build the template string for the directory name.
dir_name_template = "%03d/%03d/%5.3f/%04d/%04d/%5.3f/%06d"

# Build the template string for the run command.
cmd_template = (
    "pde1_bvp_coupled_pinn_1D.py"
    " --verbose"
    " --no-convcheck"
    " --no-save_weights"
    " --save_model"
    " --problem=%s"
    " --seed=%d"
    " --n_layers=%d"
    " --n_hid=%d"
    " --learning_rate=%g"
    " --nt_train=%d"
    " --nx_train=%d"
    " --w_bc=%g"
    " --max_epochs=%d"
)

for n_layers in n_layerss:
    for H in Hs:
        for learning_rate in learning_rates:
            for nt_train in nt_trains:
                for nx_train in nx_trains:
                    for w_bc in w_bcs:
                        for max_epochs in max_epochss:

                            # Compute the name of the directory to
                            # hold this run.
                            dir_name = dir_name_template % (
                                n_layers, H, learning_rate,
                                nt_train, nx_train,
                                w_bc, max_epochs
                            )
                            print(dir_name)

                            # Compute the random number generator seed.
                            # seed = int(time.time())
                            seed = 0

                            # Compute the command string for the run.
                            cmd = cmd_template % (
                                problem, seed,
                                n_layers, H, learning_rate,
                                nt_train, nx_train,
                                w_bc, max_epochs
                            )
                            print(cmd)

                            # Create the run directory.
                            os.makedirs(dir_name)

                            # Save the starting directory.
                            cwd = os.getcwd()

                            # Move to the run directory.
                            os.chdir(dir_name)

                            # Execute the run command, saving output.
                            os.system("%s >& run.out" % cmd)

                            # Move back to the starting directory.
                            os.chdir(cwd)
