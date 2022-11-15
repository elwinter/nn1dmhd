#!/usr/bin/env python

import os

n_layerss = [1, 2, 3, 4]
Hs = [10, 20, 40, 80]
learning_rates = [0.01]
nx_trains = [11]
ny_trains = [11]
nz_trains = [11]
nt_trains = [11]
w_bcs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
max_epochss = [1000]

# Build the template string for the directory name.
dir_name_template = "%03d/%03d/%5.3f/%04d/%04d/%04d/%04d/%5.3f/%06d"

# Save the starting directory.
cwd = os.getcwd()

# Build the template string for the run command.
cmd_template = os.path.join(cwd, "plot_run.py")

for n_layers in n_layerss:
    for H in Hs:
        for learning_rate in learning_rates:
            for nx_train in nx_trains:
                for ny_train in ny_trains:
                    for nz_train in nz_trains:
                        for nt_train in nt_trains:
                            for w_bc in w_bcs:
                                for max_epochs in max_epochss:

                                # Compute the name of the directory to hold this
                                    # run.
                                    dir_name = dir_name_template % (
                                        n_layers, H, learning_rate,
                                        nx_train, ny_train, nz_train, nt_train,
                                        w_bc, max_epochs
                                    )
                                print(dir_name)

                                # Compute the command string for the run.
                                cmd = cmd_template

                                # Move to the run directory.
                                os.chdir(dir_name)

                                # Execute the run command, saving output.
                                os.system("%s >& plot_run.out" % cmd)

                                # Move back to the starting directory.
                                os.chdir(cwd)
