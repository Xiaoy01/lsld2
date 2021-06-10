from lmfit import Parameters

from lsld2.fit.main import dummy_sat_residual_wrapper, fit

"""
Simple non-linear least squares fit. This file should be placed in the root
directory of the lsld2 repository and run from there. It utilizes the
Levenberg-Marquardt algorithm to minimize the dummy residual.

The expected result is as follows:
    scale = 1.0
    dx = 7.2
    dy = 7.2
    dz = 8

The same test can be executed for the CLI by running [cli.py] within
[lsld2/fit] and entering the sequence of commands listed below.

init_params_fit scale 0.1 dx 9.5 dy 9.5 dz 9.5
init_params_nonfit nort 0 b1 0.5 c20 0.0
min scale 0.01
max scale 1
min dx 5.1
max dx 10
min dz 5.1
max dz 10
constraint dy dx
fit_dummy levmar
print_report
"""
params = Parameters()
params.add("scale", value=0.1, min=0.01, max=1)
params.add("dx", value=9.5, min=5.1, max=10)
params.add("dz", value=9.5, min=5.1, max=10)
params.add("dy", value=9.5, expr="dx")

fit(dummy_sat_residual_wrapper, params, kws={"nort": 0, "b1": 0.5, "c20": 0},
    algo_choice="levmar")

print("Test complete.")
print("Expected Return Parameters:\n\tscale = 1.0\n\tdx = 7.2\n\tdy = 7.2"
      "\n\tdz = 8")
