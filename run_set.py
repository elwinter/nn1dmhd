#!/usr/bin/env python

import os
import time

problem = "/homes/winteel1/research/src/nn1dmhd/nn1dmhd/problems/static3.py"
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

# Build the template string for the run command.
cmd_template = (
    "pde1_bvp_coupled_pinn_3D.py"
    " --verbose"
    " --no-convcheck"
    " --no-save_weights"
    " --save_model"
    " --n_layers=%d"
    " --n_hid=%d"
    " --seed=%d"
    " --learning_rate=%g"
    " --problem=%s"
    " --max_epochs=%d"
    " --nx_train=%d"
    " --ny_train=%d"
    " --nz_train=%d"
    " --nt_train=%d"
    " --w_bc=%g"
)

for n_layers in n_layerss:
    for H in Hs:
        for learning_rate in learning_rates:
            for nx_train in nx_trains:
                for ny_train in ny_trains:
                    for nz_train in nz_trains:
                        for nt_train in nt_trains:
                            for w_bc in w_bcs:
                                for max_epochs in max_epochss:
                                    # Compute the name of the directory to
                                    # hold this run.
                                    dir_name = dir_name_template % (
                                        n_layers, H, learning_rate,
                                        nx_train, ny_train, nz_train, nt_train,
                                        w_bc, max_epochs)
                                    print(dir_name)

                                    # Compute the random number generator seed.
                                    # seed = int(time.time())
                                    seed = 0

                                    # Compute the command string for the run.
                                    cmd = cmd_template % (
                                        n_layers, H, seed, learning_rate, problem,
                                        max_epochs, nx_train, ny_train, nz_train,
                                        nt_train, w_bc
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
