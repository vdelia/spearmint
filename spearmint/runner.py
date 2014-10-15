import sys
import os
import traceback

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
        
    def __call__(self, _id, name, parameters, expt_dir):
        """Run a Python function."""
        # Add experiment directory to the system path.
        sys.path.append(os.path.realpath(expt_dir))

        k = self.cache_key(parameters)
        result = self.memoizer.get(k, None)

        if result is None:
            # Load up this module and run
            module  = __import__(name)
            result = module.main(_id, parameters)
            self.memoizer[k] = result

        else:
            logging.info("Memoized for %s: %s", k, result)

        logging.info("Got result %f", result)
        return result
