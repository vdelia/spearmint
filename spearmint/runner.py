import sys
import os
import traceback

from spearmint_pb2   import *
from ExperimentGrid  import *
from helpers         import *

import logging


# TODO: change this function to be more flexible when running python jobs
# regarding the python path, experiment directory, etc...
def run_python_job(job):
    '''Run a Python function.'''


    # Add experiment directory to the system path.
    sys.path.append(os.path.realpath(job.expt_dir))

    # Convert the PB object into useful parameters.
    params = {}
    for param in job.param:
        dbl_vals = param.dbl_val._values
        int_vals = param.int_val._values
        str_vals = param.str_val._values

        if len(dbl_vals) > 0:
            params[param.name] = np.array(dbl_vals)
        elif len(int_vals) > 0:
            params[param.name] = np.array(int_vals, dtype=int)
        elif len(str_vals) > 0:
            params[param.name] = str_vals
        else:
            raise Exception("Unknown parameter type.")

    # Load up this module and run
    module  = __import__(job.name)
    result = module.main(job.id, params)

    logging.info("Got result %f", result)

    # Store the result.
    job.value = result
    save_job(job)


