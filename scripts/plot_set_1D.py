#!/usr/bin/env python


import os
import sys


problem = sys.argv[1]
n_layerss = [1]
Hs = [10]
learning_rates = [0.01]
nt_trains = [10]
nx_trains = [20]
w_bcs = [0.95]
max_epochss = [100]

# Build the template string for the directory name.
dir_name_template = "%03d/%03d/%5.3f/%04d/%04d/%5.3f/%06d"

# Save the starting directory.
cwd = os.getcwd()

# Build the template string for the run command.
cmd_template = "plot_run_1D.py %s"

for n_layers in n_layerss:
    for H in Hs:
        for learning_rate in learning_rates:
            for nt_train in nt_trains:
                for nx_train in nx_trains:
                    for w_bc in w_bcs:
                        for max_epochs in max_epochss:

                            # Compute the name of the directory to hold this
                            # run.
                            dir_name = dir_name_template % (
                                n_layers, H, learning_rate,
                                nt_train, nx_train,
                                w_bc, max_epochs
                            )
                            print(dir_name)

                            # Move to the run directory.
                            os.chdir(dir_name)

                            # Compute the command string for the run.
                            cmd = cmd_template % problem

                            # Execute the run command, saving output.
                            os.system("%s >& plot_run.out" % cmd)

                            # Move back to the starting directory.
                            os.chdir(cwd)
