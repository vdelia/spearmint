import sys
import os
import traceback

from spearmint_pb2   import *
from ExperimentGrid  import *
from helpers         import *

import logging

# System dependent modules
DEFAULT_MODULES = [ 'packages/epd/7.1-2',
                    'packages/matlab/r2011b',
                    'mpi/openmpi/1.2.8/intel',
                    'libraries/mkl/10.0',
                    'packages/cuda/4.0',
                    ]

MCR_LOCATION = "/home/matlab/v715" # hack

logger = logging.getLogger(__file__)

def job_runner(job):
    '''This fn runs in a new process.  Now we are going to do a little
    bookkeeping and then spin off the actual job that does whatever it is we're
    trying to achieve.'''

    redirect_output(job_output_file(job))
    logging.info("Running in wrapper mode for '%s'", job.id)

    ExperimentGrid.job_running(job.expt_dir, job.id)

    # Update metadata and save the job file, which will be read by the job wrappers.
    job.start_t = int(time.time())
    job.status  = 'running'
    save_job(job)

    success    = False
    start_time = time.time()

    try:
        run_python_job(job)
        success = True
    except:
        logger.error("Problem running the job:")
        logger.error(sys.exc_info())
        logger.error(traceback.print_exc(limit=1000))

    end_time = time.time()
    duration = end_time - start_time

    # The job output is written back to the job file, so we read it back in to
    # get the results.
    job_file = job_file_for(job)
    job      = load_job(job_file)

    logging.info("Job file reloaded.")

    if not job.HasField("value"):
        logger.warning("Could not find value in output file.")
        success = False

    if success:
        logging.info("Completed successfully in %0.2f seconds. [%f]", 
                duration, job.value)

        # Update the status for this job.
        ExperimentGrid.job_complete(job.expt_dir, job.id,
                                    job.value, duration)
        job.status = 'complete'
    else:
        logging.warning("Job failed in %0.2f seconds", duration)

        # Update the experiment status for this job.
        ExperimentGrid.job_broken(job.expt_dir, job.id)
        job.status = 'broken'

    job.end_t    = int(time.time())
    job.duration = duration

    save_job(job)



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


