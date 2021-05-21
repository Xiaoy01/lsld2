import cmd
import csv
import io
import os
import re
import sys

import numpy as np
from lmfit import Parameters

from main import fit, fit_sat, dummy_sat_residual_wrapper


class FitCLI(cmd.Cmd):
    """
    Command line interface for fitting tool.

    Functions:
        do_pwd
        do_cd
        do_ls
        do_quit
        do_list_funcs
        do_list_vars

        do_init
        do_constraint
        do_relax
        do_min
        do_max
        do_bound
        do_fix
        do_vary
        do_set_param_val
        do_set_param_brute_step
        do_set_args
        do_set_kws
        do_fit

        do_print_params
        do_write_params
        do_print_report
        do_write_report

        do_init_params_nonfit
        do_init_params_fit
        do_read_params_nonfit
        do_read_params_fit
        do_set_b1_list
        do_set_weights
        do_normal_weights
        do_manual_weights
        do_set_spc_files
        do_print_b1_list
        do_print_weights
        do_print_spc_files
        do_fit_sat
        do_fit_dummy

    TODO:
        read params from json / replace csv reading with lmfit format json
            reading
        param/arg/kws element removal
    """
    # [cmd] variables, govern basic interface behavior
    intro = ('\nWelcome to the fitting tool.   '
             + 'Type help or ? to list commands.\n')
    prompt = '>> '
    # Store function, parameters, and args
    residual = None
    parameters = Parameters()
    args = None
    kws = dict()
    # [fit_sat] values
    b1_list = []
    weights = []
    spc_files = []
    normal_weights = True
    # Regular expression patterns for lifting functions and variables from
    # [.py] files
    fun_pat = re.compile(r"(?:^def )(.*?\(.*?\))(?:[:]$)",
                         re.MULTILINE | re.DOTALL)
    var_pat = re.compile(r"(?:^)([^#\s]*?)(?:[\s]?[=][^=])",
                         re.MULTILINE)
    # Store most recent fit report
    last_fit_report = ""

    def precmd(self, arg):
        """Modify initial user input."""
        return arg

    def emptyarg(self):
        """
        Define behavior on empty arg input.

        No operation is executed and the prompt waits for another input.
        """
        return False

    def default(self, line):
        """Define behavior on unrecognized input."""
        print("The command <%s> was not recognized." % line)
        print("Try typing help or ? to list commands.")
        return False

    def do_pwd(self, arg):
        """Print the working directory."""
        print(os.getcwd())

    def do_cd(self, arg):
        """
        Change the working directory.

        Takes a relative or absolute directory path as its argument.
        """
        try:
            os.chdir(arg)
        except FileNotFoundError:
            print("cd: %s: No such file or directory" % arg)
        except NotADirectoryError:
            print("cd: %s: Not a directory" % arg)

    def do_ls(self, arg):
        """
        List files and directories in the specified directory.

        Takes a relative or absolute directory path as its optional argument.
        If no such argument is given, the contents of the working directory is
        returned.
        """
        os.system("ls " + arg)

    def do_quit(self, arg):
        """Exit the command line interface, discarding any unsaved work."""
        print("Are you sure you would like to quit? "
              "Remember to save your work!")
        if input("y/n: ").lower() == "y":
            print("Thank you for using this tool")
            return True

    def do_list_funcs(self, arg):
        """
        List all functions in a .py file.

        This operation is intended for use in locating the residual function
        that will be subjected to fitting. Takes a filename as its argument.
        Note that only non-private top-level functions will be listed, and the
        target file must be in the working directory.
        """
        # Check if file exists as .py
        if not (os.path.isfile(arg)
                and arg[-3:] == ".py"
                and arg in os.listdir()):
            print("list_funcs: %s: Not a .py file" % arg)
            return False
        # Search file contents for top-level function declarations
        file_contents = open(arg, mode="r").read()
        for match in re.finditer(self.fun_pat, file_contents):
            # Don't return private methods
            if match.group(1)[:2] != "__":
                print("\t" + match.group(1))

    def do_list_vars(self, arg):
        """
        List all variables in a .py file.

        This operation is intended for use in locating the args list for use in
        fitting. Takes a filename as its argument. Note that only non-private
        top-level variables will be listed, and the target file must be in the
        working directory.
        """
        # Check if file exists as .py
        if not (os.path.isfile(arg)
                and arg[-3:] == ".py"
                and arg in os.listdir()):
            print("list_funcs: %s: Not a .py file" % arg)
            return False
        # Search file contents for top-level function declarations
        file_contents = open(arg, mode="r").read()
        for match in re.finditer(self.var_pat, file_contents):
            # Don't return private variables
            if match.group(1)[:2] != "__":
                print("\t" + match.group(1))

    def do_init(self, arg):
        """
        Define residual function and parameters for minimization.

        Takes an arbitrary number of arguments. The first argument is the name
        of the file containing the residual function, which must be in the
        working directory. The second argument is the name of the residual
        function. The remaining arguments are the names of parameters. Note
        that none of these names can contain spaces.

        Format:
        init <file> <function> <param1> <param2> ...
        """
        # Parse input and handle bad cases
        parsed = parse(arg)
        if len(parsed) < 2:
            print("init: Not enough arguments given")
            return False
        file_name = parsed[0]
        function_name = parsed[1]
        parameter_names = [(name, None, True, None, None, None, None)
                           for name in parsed[2:]]

        # Initialize [parameters] and names
        self.parameters.add_many(*parameter_names)

        # Import and set residual function
        try:
            self.residual = getattr(__import__(file_name, [function_name]),
                                    function_name)
        except AttributeError:
            print("Attribute Error")
        except Exception:  # TODO: identify specific exception types
            print("init: Function import failed")

    def do_constraint(self, arg):
        """
        Define a constraint for a parameter via expression.

        The parameter will be forced to equal/satisfy the given expression
        during fitting. Constraint expression does not need to be surrounded by
        anything, just separated from the parameter argument by a space.
        See https://lmfit.github.io/lmfit-py/constraints.html for more
        information.

        Format:
        constraint <parameter> <expression>
        """
        # Parse input and handle bad cases
        parsed = parse(arg)
        if len(parsed) < 2:
            print("constraint: Not enough arguments given")
            return False
        param = parsed[0]
        if param not in self.parameters:
            print("constraint: Parameter %s not found" % param)
            return False
        # Apply constraint
        expr = " ".join(parsed[1:])
        self.parameters[param].expr = expr

    def do_relax(self, arg):
        """Relax all parameter constraints."""
        for param in self.parameters:
            self.parameters[param].expr = None

    def do_min(self, arg):
        """
        Set a minimum value for a parameter.

        Use [inf] and [-inf] for infinite values.

        Format:
        min <parameter> <value>
        """
        # Parse input and handle bad cases
        parsed = parse(arg)
        if len(parsed) < 2:
            print("min: Not enough arguments given")
            return False
        param = parsed[0]
        if param not in self.parameters:
            print("min: Parameter %s not found" % param)
            return False
        val = np.inf if parsed[1] == "inf" \
            else -np.inf if parsed[1] == "-inf" \
            else None
        if val is None:
            try:
                val = float(parsed[1])
            except ValueError:
                print("min: Invalid minimum value")
                return False
        # Apply minimum
        self.parameters[param].min = val

    def do_max(self, arg):
        """
        Set a maximum value for a parameter.

        Use [inf] and [-inf] for infinite values.

        Format:
        max <parameter> <value>
        """
        # Parse input and handle bad cases
        parsed = parse(arg)
        if len(parsed) < 2:
            print("max: Not enough arguments given")
            return False
        param = parsed[0]
        if param not in self.parameters:
            print("max: Parameter %s not found" % param)
            return False
        val = np.inf if parsed[1] == "inf" \
            else -np.inf if parsed[1] == "-inf" \
            else None
        if val is None:
            try:
                val = float(parsed[1])
            except ValueError:
                print("max: Invalid maximum value")
                return False
        # Apply maximum
        self.parameters[param].max = val

    def do_bound(self, arg):
        """
        Set a minimum value for a parameter.

        Use [inf] and [-inf] for infinite values.

        Format:
        bound <parameter> <minimum> <maximum>
        """
        # Parse input and handle bad cases
        parsed = parse(arg)
        if len(parsed) < 3:
            print("bound: Not enough arguments given")
            return False
        param = parsed[0]
        if param not in self.parameters:
            print("bound: Parameter %s not found" % param)
            return False
        min_val = np.inf if parsed[1] == "inf" \
            else -np.inf if parsed[1] == "-inf" \
            else None
        if min_val is None:
            try:
                min_val = float(parsed[1])
            except ValueError:
                print("bound: Invalid minimum value")
                return False
        max_val = np.inf if parsed[2] == "inf" \
            else -np.inf if parsed[2] == "-inf" \
            else None
        if max_val is None:
            try:
                max_val = float(parsed[2])
            except ValueError:
                print("bound: Invalid maximum value")
                return False
        # Apply bounds
        self.parameters[param].min = min_val
        self.parameters[param].max = max_val

    def do_fix(self, arg):
        """
        Prevent fitting from varying the value of a parameter.

        Requires that the parameter already have a value.
        """
        # Handle bad inputs
        if arg == "":
            print("fix: Not enough arguments")
            return False
        if arg not in self.parameters:
            print("fix: Parameter %s not found" % arg)
            return False
        if self.parameters[arg].value is None:
            print("fix: Parameter %s requires a value before being fixed"
                  % arg)
        self.parameters[arg].vary = False

    def do_vary(self, arg):
        """
        Allow fitting to vary the value of a parameter.
        """
        # Handle bad inputs
        if arg == "":
            print("vary: Not enough arguments")
            return False
        if arg not in self.parameters:
            print("vary: Parameter %s not found" % arg)
            return False
        self.parameters[arg].vary = True

    def do_set_param_val(self, arg):
        """Set the value of a parameter."""
        # Handle bad inputs
        parsed = parse(arg)
        if len(parsed) < 2:
            print("set_param_val: Not enough arguments")
            return False
        if parsed[0] not in self.parameters:
            print("set_param_val: Parameter %s not found" % arg)
            return False
        try:
            self.parameters[parsed[0]] = float(parsed[1])
        except ValueError:
            print("set_param_val: Invalid value")
            return False

    def do_set_param_brute_step(self, arg):
        """Set the brute_step of a parameter."""
        # Handle bad inputs
        parsed = parse(arg)
        if len(parsed) < 2:
            print("set_param_brute_step: Not enough arguments")
            return False
        if parsed[0] not in self.parameters:
            print("set_param_brute_step: Parameter %s not found" % arg)
            return False
        try:
            self.parameters[parsed[0]] = float(parsed[1])
        except ValueError:
            print("set_param_brute_step: Invalid value")
            return False

    def do_set_args(self, arg):
        """
        Define args to be passed into residual function during minimization.

        The first argument is the name of the file containing the args list,
        which must be in the working directory. The second argument is the name
        of the args list. Note that neither of these names can contain spaces.

        Format:
        set_args <file> <args>
        """
        # Parse input and handle bad cases
        parsed = parse(arg)
        if len(parsed) < 2:
            print("set_args: Not enough arguments given")
            return False
        file_name = parsed[0]
        list_name = parsed[1]

        # Import and set args list
        try:
            self.args = getattr(__import__(file_name, [list_name]),
                                list_name)
        except Exception:  # TODO: identify specific exception types
            print("set_args: Variable import failed")

    def do_set_kws(self, arg):
        """
        Define kws to be passed into residual function during minimization.

        The first argument is the name of the file containing the kws dict,
        which must be in the working directory. The second argument is the name
        of the kws dict. Note that neither of these names can contain spaces.

        Format:
        set_args <file> <args>
        """
        # Parse input and handle bad cases
        parsed = parse(arg)
        if len(parsed) < 2:
            print("set_args: Not enough arguments given")
            return False
        file_name = parsed[0]
        dict_name = parsed[1]

        # Import and set args list
        try:
            self.kws = getattr(__import__(file_name, [dict_name]),
                               dict_name)
        except Exception:  # TODO: identify specific exception types
            print("set_kws: Variable import failed")

    def do_fit(self, arg):
        """
        Fit residual function using selected algorithm and optional keyword
        arguments.


        The first argument, <algo_choice>, is the choice of fitting algorithm,
        which can be simplex, levmar, mcmc, grid, montecarlo, or genetic. The
        second argument, <fit_kws>, (optional) is a dictionary of keyword
        arguments to be passed into the selected fitting algorithm as
        [**kwargs]. See [main.py] for specific documentation on what keyword
        arguments can be used.

        <fit_kws> should be entered with full Python syntax, as it will be
        interpreted and evaluated as python code before being passed on to the
        fitting algorithm.

        Format:
        fit <algo_choice> <fit_kws>
        """
        # Parse input
        parsed = parse(arg)
        # Switch output channel to suppress and store printing
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        # Execute fit
        estring = " ".join(parsed[1:])
        fit_kws = dict()
        if estring != "":
            fit_kws = eval(" ".join(parsed[1:]))
        fit(self.residual, self.parameters, args=self.args, kws=self.kws,
            algo_choice=parsed[0], **fit_kws)
        # Record output and return to original output channel
        self.last_fit_report = new_stdout.getvalue()
        sys.stdout = old_stdout

    def do_print_params(self, arg):
        """Print parameters."""
        self.parameters.pretty_print()

    def do_write_params(self, arg):
        """
        Write parameters to the specified file.

        Format: write_params <filename>
        """
        # Store original output channel
        old_stdout = sys.stdout
        # Open file and write
        with open(arg, "w") as f:
            sys.stdout = f
            self.parameters.pretty_print()
        # Restore original output channel
        sys.stdout = old_stdout

    def do_print_report(self, arg):
        """Print a fitting report."""
        print(self.last_fit_report)

    def do_write_report(self, arg):
        """
        Write a fitting report to the specified file.

        Format: write_report <filename>
        """
        # Store original output channel
        old_stdout = sys.stdout
        # Open file and write
        with open(arg, "w") as f:
            sys.stdout = f
            print(self.last_fit_report)
        # Restore original output channel
        sys.stdout = old_stdout

    def do_init_params_nonfit(self, args):
        """
        Manually enter nonfit parameters, aka kws. Values can be any type,
        but should be written with python syntax.

        Format: init_params_nonfit <name1> <val1> ... <namen> <valn>
        """
        # Parse input and handle bad cases
        parsed = parse(args)
        if len(parsed) < 2:
            print("init_params_nonfit: Not enough arguments given")
            return False
        if len(parsed) % 2 == 1:
            print("init_params_nonfit: Parameter given without value")
            return False
        try:
            for i in range(0, len(parsed), 2):
                self.kws[parsed[i]] = eval(parsed[i + 1])
        except SyntaxError:
            print("init_params_nonfit: Invalid value supplied")
            return False

    def do_init_params_fit(self, args):
        """
        Manually enter fit parameters. Values must be numeric.

        Format: init_params_fit <name1> <val1> ... <namen> <valn>
        """
        # Parse input and handle bad cases
        parsed = parse(args)
        if len(parsed) < 2:
            print("init_params_fit: Not enough arguments given")
            return False
        if len(parsed) % 2 == 1:
            print("init_params_fit: Parameter given without value")
            return False
        try:
            for i in range(0, len(parsed), 2):
                self.parameters.add(parsed[i], value=float(parsed[i + 1]))
        except ValueError:
            print("init_params_fit: Non-numeric value supplied")
            return False

    def do_read_params_nonfit(self, args):
        """
        Read nonfit parameters, aka kws, from a csv file.

        Format: read_params_nonfit <file>
        """
        # Read csv file
        try:
            with open(args, newline="") as csvfile:
                reader = csv.reader(csvfile)
                # Set values
                for row in reader:
                    try:
                        self.kws[row[0]] = eval(row[1])
                    except SyntaxError:
                        print("read_params_nonfit: Invalid value supplied")
                        return False
        except FileNotFoundError:
            print("read_params_nonfit: Target file not found")
            return False

    def do_read_params_fit(self, args):
        """
        Read fit parameters for fit_sat from a csv file.

        Format: read_params_fit <file>
        """
        # Read csv file
        try:
            with open(args, newline="") as csvfile:
                reader = csv.reader(csvfile)
                # Set values
                for row in reader:
                    try:
                        self.parameters.add(row[0], value=float(row[1]))
                    except ValueError:
                        print("read_params_fit: Non-numeric value supplied")
                        return False
        except FileNotFoundError:
            print("read_params_fit: Target file not found")
            return False

    # set_b1_list <val> ... <val>
    def do_set_b1_list(self, args):
        """Set the b1_list values for fit_sat."""
        parsed = parse(args)
        try:
            self.b1_list = [float(str) for str in parsed]
        except ValueError:
            print("set_b1_list: Non-numeric value supplied")
            return False

    # set_weights <val> ... <val>
    def do_set_weights(self, args):
        """Set the weight values for fit_sat."""
        parsed = parse(args)
        try:
            self.weights = [float(str) for str in parsed]
        except ValueError:
            print("set_weights: Non-numeric value supplied")
            return False

    # normal_weights
    def do_normal_weights(self, args):
        """
        Sets weights to normalize experimental data for fit_sat.
        Does not take effect until fit_sat is called.
        """
        self.normal_weights = True

    # manual_weights
    def do_manual_weights(self, args):
        """Use manually entered weights on experimental data fit_sat."""
        self.normal_weights = False

    # set_spc_files <val> ... <val>
    def do_set_spc_files(self, args):
        """Set the spc files for fit_sat."""
        self.spc_files = parse(args)

    # print_b1_list
    def do_print_b1_list(self, arg):
        """Print b1_list."""
        print(self.b1_list)

    # print_weights
    def do_print_weights(self, arg):
        """Print weights."""
        print(self.weights)

    # print_spc_files
    def do_print_spc_files(self, arg):
        """Print weights."""
        print(self.spc_files)

    # fit_sat <algo_choice> <fit_kws>
    def do_fit_sat(self, args):
        """
        Fit sat_residual using selected algorithm and optional keyword
        arguments.


        The first argument, <algo_choice>, is the choice of fitting algorithm,
        which can be simplex, levmar, mcmc, grid, montecarlo, or genetic. The
        second argument, <fit_kws>, (optional) is a dictionary of keyword
        arguments to be passed into the selected fitting algorithm as
        [**kwargs]. See [main.py] for specific documentation on what keyword
        arguments can be used.

        <fit_kws> should be entered with full Python syntax, as it will be
        interpreted and evaluated as python code before being passed on to the
        fitting algorithm.

        Format:
        fit_sat <algo_choice> <fit_kws>
        """
        # Parse input
        parsed = parse(args)
        if len(parsed) < 1:
            print("fit_sat: Not enough arguments given")
            return False
        algo_choice = parsed[0]
        estring = " ".join(parsed[1:])
        fit_kws = dict()
        if estring != "":
            fit_kws = eval(" ".join(parsed[1:]))

        # Verify size compatibility between [b1_list], [weights],
        # and [spc_files]
        if not ((len(self.b1_list) == len(self.weights) or self.normal_weights)
                and len(self.b1_list) == len(self.spc_files)):
            print("fit_sat: Incompatible input sizes:\n")
            print("b1_list has %d elements" % len(self.b1_list))
            print("weights has %d elements" % len(self.weights))
            print("spc_files has %d elements" % len(self.spc_files))
            return False

        # Read in [spc_files]
        bgrid = [[] for i in range(len(self.spc_files))]
        spec_expt = [[] for i in range(len(self.spc_files))]
        try:
            # Iterate over each filename
            for i in range(len(self.spc_files)):
                # Read file
                with open(self.spc_files[i], newline="") as csvfile:
                    reader = csv.reader(csvfile)
                    # Write file contents
                    for row in reader:
                        try:
                            bgrid[i].append(float(row[0]))
                            spec_expt[i].append(float(row[1]))
                        except ValueError:
                            print("fit_sat: Non-numeric value supplied")
                            return False
        except FileNotFoundError:
            print("fit_sat: spc file not found")
            return False
        # Catch ragged array and convert to numpy ndarray
        if (any(len(col) != len(bgrid[0]) for col in bgrid)
                or any(len(col) != len(spec_expt[0]) for col in spec_expt)):
            print("fit_sat: Inconsistent spc file sizes")
            return False
        bgrid = np.array(bgrid)
        spec_expt = np.array(spec_expt)

        # Set normal weights if necessary
        if self.normal_weights:
            self.weights = list(np.max(spec_expt, axis=1)
                                - np.min(spec_expt, axis=1))

        # Switch output channel to suppress and store printing
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout

        # Execute fit
        fit_sat(params_fit=self.parameters,
                params_nonfit=self.kws,
                bgrid=bgrid,
                spec_expt=spec_expt,
                b1_list=self.b1_list,
                weights=self.weights,
                algo_choice=algo_choice, **fit_kws)

        # Record output and return to original output channel
        self.last_fit_report = new_stdout.getvalue()
        sys.stdout = old_stdout

    def do_fit_dummy(self, args):
        """
        Sanity check, fits [dummy_sat_residual_wrapper].

        Initiate with the following
        parameters:
            scale = 0.1, min 0.01, max 1
            dx = 9.5, min 5.1, max 10
            dy, expr = dx
            dz = 9.5, min 5.1, max 10
        kws:
            nort = 0
            b1 = 0.5
            c20 = 0.0
        Expect to return
        parameters:
            scale = 1.0
            dx = 7.2
            dy = 7.2
            dz = 8

        Format:
        fit_sat <algo_choice> <fit_kws>
        """
        # Parse input
        parsed = parse(args)
        if len(parsed) < 1:
            print("fit_dummy: Not enough arguments given")
            return False
        algo_choice = parsed[0]
        estring = " ".join(parsed[1:])
        fit_kws = dict()
        if estring != "":
            fit_kws = eval(" ".join(parsed[1:]))

        # Verify size compatibility between [b1_list], [weights],
        # and [spc_files]
        if not ((len(self.b1_list) == len(self.weights) or self.normal_weights)
                and len(self.b1_list) == len(self.spc_files)):
            print("fit_sat: Incompatible input sizes:\n")
            print("b1_list has %d elements" % len(self.b1_list))
            print("weights has %d elements" % len(self.weights))
            print("spc_files has %d elements" % len(self.spc_files))
            return False

        # Switch output channel to suppress and store printing
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout

        # Execute fit
        fit(dummy_sat_residual_wrapper, self.parameters, args=self.args,
            kws=self.kws, algo_choice=algo_choice, **fit_kws)

        # Record output and return to original output channel
        self.last_fit_report = new_stdout.getvalue()
        sys.stdout = old_stdout


def parse(arg):
    """Convert a series of zero or more arguments to a string tuple."""
    return tuple(arg.split())


if __name__ == '__main__':
    FitCLI().cmdloop()
