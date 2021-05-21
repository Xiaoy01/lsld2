import io
from random import getrandbits
import sys

from lmfit import minimize, report_fit, Parameters
import numpy as np

from .saturation_calc import sat_residual

# Lists of fitting algorithms, separated by whether they are custom
# implementations or wrappers of lmfit functions
lmfit_algos = ["simplex", "levmar", "mcmc", "grid"]
custom_algos = ["montecarlo", "genetic"]


def fit_sat(params_fit=Parameters(),
            params_nonfit={'shiftg': -3.9, 'nort': 20},
            bgrid=np.reshape(np.linspace(-60, 60, 256) + 3360, (1, -1)),
            spec_expt=np.zeros((1, 128)), b1_list=[0.1], weights=[1],
            algo_choice="simplex", **fit_kws):
    """
    Minimize the [sat_residual()] function using the specified
    arguments and algorithm.

    Calls on the [fit()] function to do so.

    Args:
        params_fit (dict, optional): Parameters to be varied and their initial
            values.
        params_nonfit (dict, optional): Parameters to be held constant and
            their associated values.
        bgrid (np.ndarray, optional): Grid of magnetic field values in Gauss,
            need not be uniformly spaced.
        spec_expt (np.ndarray, optional): Experimental data used to calculate
            the residual.
        b1_list (list, optional): List of b1 values associated with rows of
            [bgrid] and [spec_expt].
        weights (list, optional): List of weights to be applied to residuals.
        algo_choice (str, optional): Name of algorithm used for minimization.
            Possible values are as follows: "simplex", "levmar", "mcmc",
            "grid", "montecarlo", or "genetic".
        **fit_kws (dict, optional): Keyword arguments to be passed into the
            chosen fitting algorithm. "montecarlo" will require argument
            "montecarlo_trial_count" to dictate how many trials it conducts.
            "genetic" will require arguments "genetic_generation_count",
            "genetic_generation_size", and "genetic_survival_rate" to dictate
            the number of generations produced, the number of trials in each
            generation, and how many trials in each generation will be kept for
            the next generation, respectively. Other fitting algorithms will
            require arguments as documented by [lmfit].

    Returns:
        A mapping of parameters in [params] to the optimal values computed by
        the selected fitting algorithm.
    Return type:
        [lmfit.Parameters]
    """
    # Compile keyword arguments
    params_nonfit["bgrid"] = bgrid
    params_nonfit["spec_expt"] = spec_expt
    params_nonfit["b1_list"] = b1_list
    params_nonfit["weights"] = weights

    # Run fitting
    return fit(sat_residual_wrapper, params_fit, kws=params_nonfit,
               algo_choice=algo_choice, **fit_kws)


def sat_residual_wrapper(params, **kws):
    """
    [sat_residual()] restructured to conform with constraints
    of [lmfit] residual functions.

    See [sat_residual()] for full documentation of arguments.

    Args:
        params (lmfit.Parameters): all bindings that would be stored in
            [params_fit] for [sat_residual()]
        **kws (dict, optional): all bindings that would be stored in
            [params_nonfit] for [sat_residual()], as well as
            all other optional arguments of [sat_residual()].

    Returns:
        Residual values computed by [sat_residual()].
    Return type:
        [numpy.ndarray]
    """
    params_fit = dict(params.valuesdict())
    params_nonfit = kws
    bgrid = params_nonfit.pop("bgrid")
    spec_expt = params_nonfit.pop("spec_expt")
    b1_list = params_nonfit.pop("b1_list")
    weights = params_nonfit.pop("weights")

    # Get output
    return sat_residual(params_fit=params_fit,
                        params_nonfit=params_nonfit,
                        bgrid=bgrid,
                        spec_expt=spec_expt,
                        b1_list=b1_list,
                        weights=weights)


def dummy_sat_residual_wrapper(params, **kws):
    """
    Toy version of [sat_residual_wrapper()] to be used for sanity checking.

    Args:
        params (lmfit.Parameters): bindings for floats "scale", "dx", "dy", and
            "dz". Optimal value bindings are {"scale": 1.0, "dx": 7.2,
            "dy": 7.2, "dz": 8}.
        **kws (dict, optional): bindings for floats "nort", "b1", and "c20".
            Best set to {"nort": 0, "b1": 0.5, "c20": 0}.

    Returns:
        Residual values.
    Return type:
        [numpy.ndarray]
    """
    bgrid = np.linspace(-60, 60, 256) + 3360
    # Construct fake data
    spec_expt = saturation_calc.cw_spec(bgrid=bgrid,
                                        params_in={**kws, **{'scale': 1.0,
                                                             'dx': 7.2,
                                                             'dy': 7.2,
                                                             'dz': 8}},
                                        basis_file='xoxo', prune_on=0)[1]
    # Construct the same model as [sat_residual_wrapper()], but with simple
    # arguments
    spec_simulated = saturation_calc.cw_spec(bgrid=bgrid,
                                             params_in={**kws,
                                                        **{x: params[x].value
                                                           for x in params}},
                                             basis_file='xoxo',
                                             prune_on=False)[1]
    # Return residual value
    return spec_expt - spec_simulated


def fit(residual_function, params, args=None, kws=None, algo_choice="simplex",
        **fit_kws):
    """
    Minimize a residual function using the specified algorithm.

    Args:
        residual_function (function): The function to minimize. Must have a
            signature compatible with the [lmfit] requirements,
            [residual_function(params, *args, **kws)]. See [lmfit] for more
            details.
        params (lmfit.Parameters): Information on parameters to be passed into
            [residual_function()]. Controls initial value, constraints on
            potential values, whether or not to vary each parameter, and how to
            do so. "montecarlo" and "genetic" fitting algorithms require [min]
            and [max] values for parameters. Other fitting algorithms have
            requirements as documented by [lmfit].
        args (tuple, optional): Positional argument values to be passed into
            [residual_function]. These values will not be varied as a part of
            the fitting process.
        kws (list, optional): Keyword argument values to be passed into
            [residual_function]. These values will not be varied as a part of
            the fitting process.
        algo_choice (str, optional): Name of algorithm used for minimization.
            Possible values are as follows: "simplex", "levmar", "mcmc",
            "grid", "montecarlo", or "genetic".
        **fit_kws (dict, optional): Keyword arguments to be passed into the
            chosen fitting algorithm. "montecarlo" will require argument
            "montecarlo_trial_count" to dictate how many trials it conducts.
            "genetic" will require arguments "genetic_generation_count",
            "genetic_generation_size", and "genetic_survival_rate" to dictate
            the number of generations produced, the number of trials in each
            generation, and how many trials in each generation will be kept for
            the next generation, respectively. Other fitting algorithms will
            require arguments as documented by [lmfit].

    Returns:
        A mapping of parameters in [params] to the optimal values computed by
        the selected fitting algorithm.
    Return type:
        [lmfit.Parameters]
    """
    # Differentiate between custom fitting and [lmfit] fitting
    if algo_choice in custom_algos:
        return __custom_fit(residual_function, params, algo_choice, args=args,
                            kws=kws, **fit_kws)
    elif algo_choice in lmfit_algos:
        return __lmfit_fit(residual_function, params, algo_choice, args=args,
                           kws=kws, **fit_kws)
    else:
        raise ValueError("algo_choice invalid")


def __lmfit_fit(residual_function, params, algo_choice, args=None, kws=None,
                **fit_kws):
    """Process calls for [lmfit] fitting."""
    method = "nelder" if algo_choice == "simplex" else \
        "leastsq" if algo_choice == "levmar" else \
        "emcee" if algo_choice == "mcmc" else \
        "brute" if algo_choice == "grid" else None
    if method is None:
        raise ValueError("algo_choice invalid")

    # Switch output channel to suppress printing during fitting process
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    # Call [lmfit.minimize()]
    out = minimize(residual_function, params, method=method, nan_policy='omit',
                   args=args, kws=kws, **fit_kws)
    # Return to original output channel
    sys.stdout = old_stdout
    # Print report of fitting results
    print(report_fit(out.params))
    # Return value bindings
    return out


def __custom_fit(residual_function, params, algo_choice, args=None, kws=None,
                 **fit_kws):
    """Process calls for custom fitting."""
    # Handle NoneType arguments
    args = [] if args is None else args
    kws = {} if kws is None else kws
    # Run fitting according to method selection
    # Switch output channel to suppress printing during fitting process
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    # Call relevant custom fitting function
    if algo_choice == "montecarlo":
        out = __montecarlo_fit(residual_function, params, args, kws,
                               fit_kws["montecarlo_trial_count"]
                               if "montecarlo_trial_count" in fit_kws
                               else 1000)
    elif algo_choice == "genetic":
        out = __genetic_fit(residual_function, params, args, kws,
                            fit_kws["genetic_generation_count"]
                            if "genetic_generation_count" in fit_kws
                            else 100,
                            fit_kws["genetic_generation_size"]
                            if "genetic_generation_size" in fit_kws
                            else 100,
                            fit_kws["genetic_survival_rate"]
                            if "genetic_survival_rate" in fit_kws
                            else 0.2)
    else:
        raise ValueError("algo_choice invalid")
    # Return to original output channel
    sys.stdout = old_stdout
    # Print report of fitting results
    print("[[Variables]]")
    for param in params:
        val = out[param].value
        ini = params[param].value
        print("    {0:7.7}{1:0<11.11} (init = {2})".format(
            param + ":", (" " if val >= 0 else "") + str(val), ini))
    # Return value bindings
    return out


def __montecarlo_fit(residual_function, params, args, kws, trial_count):
    """Monte carlo fitting."""
    # Record minimum residual sum of squares and corresponding parameters
    min_params = params.copy()
    min_rss = __rss(residual_function, min_params, args, kws)

    # Generate random parameter sets and return optimal set
    for trial in range(trial_count):
        # Generate new parameters uniformly
        trial_params = __random_param_values(params)
        # Get corresponding residual value of new parameters
        trial_rss = __rss(residual_function, trial_params, args, kws)
        # Record new parameters and rss if better
        if trial_rss < min_rss:
            min_rss = trial_rss
            min_params = trial_params

    # Return best parameters
    return min_params


def __genetic_fit(residual_function, params, args, kws, generation_count,
                  generation_size, survival_rate):
    """Genetic algorithm fitting."""
    # FLAW: CANNOT MUTATE PARAMETER FROM ONE SIGN TO ANOTHER
    # Uses rss as fitness to minimize
    # Unclear if fitness values will have sufficient range
    def new_generation(previous_population=None, previous_fitnesses=None):
        # Generate original population
        if previous_population is None or previous_fitnesses is None:
            population = [__random_param_values(params)
                          for i in range(generation_size)]
        # Generate descendant population
        else:
            # Use normalized fitness to calculate survival and breeding odds
            normfit = -1 * (previous_fitnesses - np.amax(previous_fitnesses))
            normfit = normfit / np.sum(normfit)
            survivor_count = int(survival_rate * generation_size)
            children_count = generation_size - survivor_count
            # Get survivors and parents
            survivors = list(np.random.choice(previous_population,
                                              size=survivor_count, p=normfit,
                                              replace=False))
            parents = list(np.random.choice(previous_population,
                                            size=children_count * 2,
                                            p=normfit))
            # Produce children via crossover between parents, then mutate
            children = [parents[i * 2].copy() for i in range(children_count)]
            for i in range(0, children_count):
                for param in params:
                    # Verify that param can be modified
                    if params[param].vary:
                        # Crossover
                        children[i][param] = (children[i][param]
                                              if bool(getrandbits(1))
                                              else parents[2 * i + 1][param])
                        # Mutation
                        children[i][param].value = (children[i][param].value
                                                    * np.random.uniform(0.75,
                                                                        1.25))
            population = survivors + children
        # Calculate fitnesses
        fitnesses = np.array([__rss(residual_function, chromosome, args, kws)
                              for chromosome in population])
        # Return
        return population, fitnesses

    # Store generation and fitnesses
    population = fitnesses = None
    # Main generation loop
    for generation in range(generation_count):
        population, fitnesses = new_generation(population, fitnesses)
    # Return best parameters from final generation
    return population[np.argmin(fitnesses)]


def __random_param_values(params):
    """Generate new parameters uniformly."""
    rand_params = params.copy()
    for param in params:
        # Only modify parameter if [vary == True]
        if params[param].vary:
            rand_params[param].value = np.random.uniform(params[param].min,
                                                         params[param].max)
    # Explicitly re-apply parameter [expr] constraints
    rand_params.update_constraints()
    return rand_params


def __rss(residual_function, param_values, args, kws):
    """Compute residual sum of squares"""
    return np.sum(np.square(np.array(residual_function(param_values,
                                                       *args,
                                                       **kws))))
