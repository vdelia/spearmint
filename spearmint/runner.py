import sys
import os
import traceback

from spearmint.ExperimentGrid  import *
from spearmint.helpers         import *

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
        
    def __call__(self, _id, objective_function, parameters, working_directory):
        """Run a Python function."""
        # Add experiment directory to the system path.
        sys.path.append(os.path.realpath(working_directory))

        k = self.cache_key(parameters)
        result = self.memoizer.get(k, None)
        usememoized = False

        if result is None:
            # Load up this module and run
            result = objective_function(_id, parameters)
            self.memoizer[k] = result

        else:
            usememoized = True
            logging.info("Memoized for %s: %s", k, result)

        logging.info("Got result %f", result)
        return result, usememoized

