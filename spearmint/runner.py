import sys
import os
import traceback

from spearmint_pb2   import *
from ExperimentGrid  import *
from helpers         import *

import logging
import operator

# TODO: change this function to be more flexible when running python jobs
# regarding the python path, experiment directory, etc...
class PythonRunner():
    
    def __init__(self):
        self.memoizer = {}

    def cache_key(self, params):
        items = [ (k, tuple(v)) for k,v in params.items()]
        items.sort(key=operator.itemgetter(0))
        k = tuple(items)
        return k
        
    def __call__(self, job):
        """Run a Python function."""
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

        k = self.cache_key(params)
        result = self.memoizer.get(k, None)

        if result is None:
            # Load up this module and run
            module  = __import__(job.name)
            result = module.main(job.id, params)
            self.memoizer[k] = result

        else:
            logging.info("Memoized for %s: %s", k, result)

        logging.info("Got result %f", result)

        # Store the result.
        job.value = result
