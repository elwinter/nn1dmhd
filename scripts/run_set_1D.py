#!/usr/bin/env python

import os
import time

problem = "/homes/winteel1/research/src/nn1dmhd/problems/static1.py"
n_layerss = [1, 2, 3, 4]
Hs = [10, 20, 40, 80]
learning_rates = [0.01]
nx_trains = [11]
nt_trains = [11]
w_bcs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
max_epochss = [1000]

# Build the template string for the directory name.
dir_name_template = "%03d/%03d/%5.3f/%04d/%04d/%5.3f/%06d"

# Build the template string for the run command.
cmd_template = (
    "pde1_bvp_coupled_pinn_1D.py"
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
    " --nt_train=%d"
    " --w_bc=%g"
)

for n_layers in n_layerss:
    # print("n_layers = %s" % n_layers)
    for H in Hs:
        # print("H = %s" % H)
        for learning_rate in learning_rates:
            # print("learning_rate = %s" % learning_rate)
            for nx_train in nx_trains:
                # print("nx_train = %s" % nx_train)
                For nt_train in nt_trains:
                    # print("nt_train = %s" % nt_train)
                    for w_bc in w_bcs:
                        # print("w_bc = %s" % w_bc)
                        for max_epochs in max_epochss:
                            # print("max_epochs = %s" % max_epochs)
                            # Compute the name of the directory to
                            # hold this run.
                            dir_name = dir_name_template % (
                                n_layers, H, learning_rate,
                                nx_train, nt_train, w_bc, max_epochs)
                            print(dir_name)

                            # Compute the random number generator seed.
                            # seed = int(time.time())
                            seed = 0

                            # Compute the command string for the run.
                            cmd = cmd_template % (
                                n_layers, H, seed, learning_rate, problem,
                                max_epochs, nx_train, nt_train, w_bc
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
