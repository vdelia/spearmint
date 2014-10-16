# Copyright (C) 2012 Jasper Snoek, Hugo Larochelle and Ryan P. Adams
#
# This code is written for research and educational purposes only to
# supplement the paper entitled
# "Practical Bayesian Optimization of Machine Learning Algorithms"
# by Snoek, Larochelle and Adams
# Advances in Neural Information Processing Systems, 2012
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import optparse
import importlib
import time
import imp
import os
import sys
import logging

try: import simplejson as json
except ImportError: import json


from ExperimentGrid  import *
from helpers         import *
from runner          import PythonRunner


# There are two things going on here.  There are "experiments", which are
# large-scale things that live in a directory and in this case correspond
# to the task of minimizing a complicated function.  These experiments
# contain "jobs" which are individual function evaluations.  The set of
# all possible jobs, regardless of whether they have been run or not, is
# the "grid".  This grid is managed by an instance of the class
# ExperimentGrid.
#
# The spearmint.py script can run in two modes, which reflect experiments
# vs jobs.  When run with the --run-job argument, it will try to run a
# single job.  This is not meant to be run by hand, but is intended to be
# run by a job queueing system.  Without this argument, it runs in its main
# controller mode, which determines the jobs that should be executed and
# submits them to the queueing system.


def parse_args():
    parser = optparse.OptionParser(usage="\n\tspearmint [options] <experiment/config.pb>")

    parser.add_option("--max-finished-jobs", dest="max_finished_jobs",
                      type="int", default=10000)
    parser.add_option("--method", dest="chooser_module",
                      help="Method for choosing experiments [SequentialChooser, RandomChooser, GPEIOptChooser, GPEIOptChooser, GPEIperSecChooser, GPEIChooser]",
                      type="string", default="GPEIOptChooser")
    parser.add_option("--method-args", dest="chooser_args",
                      help="Arguments to pass to chooser module.",
                      type="string", default="")
    parser.add_option("--grid-size", dest="grid_size",
                      help="Number of experiments in initial grid.",
                      type="int", default=20000)
    parser.add_option("--grid-seed", dest="grid_seed",
                      help="The seed used to initialize initial grid.",
                      type="int", default=1)
    parser.add_option("-v", "--verbose", action="store_true",
                      help="Print verbose debug output.")

    (options, args) = parser.parse_args()

    if len(args) == 0:
        parser.print_help()
        sys.exit(0)

    return options, args

def main():
    (options, args) = parse_args()

    experiment_config = args[0]
    expt_dir  = os.path.dirname(os.path.realpath(experiment_config))
    logging.info("Using experiment configuration: " + experiment_config)
    logging.info("experiment dir: " + expt_dir)

    if not os.path.exists(expt_dir):
        logging.info("Cannot find experiment directory '%s'. "
            "Aborting." % (expt_dir))
        sys.exit(-1)

    check_experiment_dirs(expt_dir)

    # Load up the chooser module.
    module  = importlib.import_module('chooser.' + options.chooser_module)
    chooser = module.init(expt_dir, options.chooser_args)

    experiment = load_experiment(experiment_config)
    optimize(experiment, expt_dir, chooser, options)

def optimize(experiment, working_directory, chooser, options):
    # Loop until we run out of jobs.
    for current_best, best_job, best_params, _next in \
            explore_space_of_candidates(experiment, 
                    working_directory, 
                    chooser, 
                    grid_size=options.grid_size,
                    grid_seed=options.grid_seed,
                    max_finished_jobs=options.max_finished_jobs):
        # This is polling frequency. A higher frequency means that the algorithm
        # picks up results more quickly after they finish, but also significantly
        # increases overhead.
        print "looping", current_best, best_job, best_params, _next


def explore_space_of_candidates(experiment, expt_dir, chooser,
        grid_size=1000,
        grid_seed=1,
        max_finished_jobs=100):
   
    # Build the experiment grid.
    expt_grid = ExperimentGrid(expt_dir,
                               experiment.variables, grid_size, grid_seed)

    next_jobid = 0
    run_python_job = PythonRunner()

    while next_jobid < max_finished_jobs:
        best_val, best_job = expt_grid.get_best()
 
        # Gets you everything - NaN for unknown values & durations.
        grid, values, durations = expt_grid.get_grid()


        # Returns lists of indices.
        candidates = expt_grid.get_candidates()
        pending    = expt_grid.get_pending()
        complete   = expt_grid.get_complete()

        n_candidates = candidates.shape[0]
        n_pending    = pending.shape[0]
        n_complete   = complete.shape[0]
        logging.info("%d candidates   %d pending   %d complete", n_candidates, 
                n_pending, n_complete)

        if n_candidates == 0:
            logging.info("There are no candidates left.  Exiting.")
            return

        # Ask the chooser to pick the next candidate
        logging.info("Choosing next candidate... ")
        job_id = chooser.next(grid, values, durations, candidates, pending, complete)

        yield best_val, best_job, expt_grid.get_params(best_job), job_id

        # If the job_id is a tuple, then the chooser picked a new job.
        # We have to add this to our grid
        if isinstance(job_id, tuple):
            (job_id, candidate) = job_id
            job_id = expt_grid.add_to_grid(candidate)

        logging.info("selected job %d from the grid", job_id)

        expt_grid.set_submitted(job_id, next_jobid)
        expt_grid.set_running(job_id)

        start_t = time.time()
        result = run_python_job(job_id, experiment.name,
                expt_grid.get_params(job_id), expt_dir)
        duration = time.time() - start_t
        expt_grid.set_complete(job_id, result, duration)

        next_jobid += 1

def check_experiment_dirs(expt_dir):
    '''Make output and jobs sub directories.'''

    output_subdir = os.path.join(expt_dir, 'output')
    check_dir(output_subdir)

    job_subdir = os.path.join(expt_dir, 'jobs')
    check_dir(job_subdir)

if __name__=='__main__':
    FORMAT = '%(asctime)-15s %(process)d %(module)s %(levelname)s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    main()

