#!/usr/bin/env python

import os
import re

n_layerss = [1, 2, 3, 4]
Hs = [10, 20, 40, 80]
learning_rates = [0.01]
nx_trains = [11]
nt_trains = [11]
w_bcs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
max_epochss = [1000]

# Build the template string for the directory name.
dir_name_template = "%03d/%03d/%5.3f/%04d/%04d/%5.3f/%06d"

# Save the starting directory.
cwd = os.getcwd()

for n_layers in n_layerss:
    for H in Hs:
        for learning_rate in learning_rates:
            for nx_train in nx_trains:
                for nt_train in nt_trains:
                    for w_bc in w_bcs:
                        for max_epochs in max_epochss:

                            # Compute the name of the directory to hold this run.
                            dir_name = dir_name_template % (
                                n_layers, H, learning_rate,
                                nx_train, nt_train, w_bc, max_epochs
                            )

                            # Move to the run directory.
                            os.chdir(dir_name)

                            # Read the run log.
                            with open("run.out", "r") as f:
                                lines = f.readlines()

                            # Extract the run end time.
                            line = lines[-7]
                            line_pattern = "Training stopped at (.+)"
                            m = re.match(line_pattern, line)
                            run_end_time = m.groups()[0]

                            # Extract the run elapsed time.
                            line = lines[-6]
                            line_pattern = "Total training time was (.+) seconds."
                            m = re.match(line_pattern, line)
                            run_elapsed_time = m.groups()[0]

                            # Extract the final values of the total, equation, and BC loss functions.
                            line = lines[-8]
                            line_pattern = 'Ending epoch \d+, \(L, L_all, L_bc\) = \((.+), (.+), (.+)\)'
                            m = re.match(line_pattern, line)
                            (L, L_all, L_bc) = m.groups()[:]

                            # Print the summary line.
                            print("%s,mollie,float32,%d,%d,0.01,%d,%d,%g,%d,%s,%s,%s,%s" %
                                (run_end_time, n_layers, H, nx_train, nt_train, w_bc,
                                max_epochs, run_elapsed_time, L, L_all, L_bc))

                            # Move back to the starting directory.
                            os.chdir(cwd)
