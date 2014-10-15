import os
import sys
import subprocess
import tempfile

from google.protobuf import text_format
from spearmint_pb2   import *
import imp

def sh(cmd):
    '''Run a shell command (blocking until completion).'''
    subprocess.check_call(cmd, shell=True)


def check_dir(path):
    '''Create a directory if it doesn't exist.'''
    if not os.path.exists(path):
        os.mkdir(path)


def grid_for(job):
    return os.path.join(job.expt_dir, 'expt-grid.pkl')



def file_write_safe(path, data):
    '''Write data to a temporary file, then move to the destination path.'''
    fh = tempfile.NamedTemporaryFile(mode='w', delete=False)
    fh.write(data)
    fh.close()
    cmd = 'mv "%s" "%s"' % (fh.name, path)
    sh(cmd)


def load_experiment(filename):
    expt = imp.load_source("_experiment_settings_", filename)
    return expt


def job_file_for(job):
    '''Get the path to the job file corresponding to a job object.'''
    return os.path.join(job.expt_dir, 'jobs', '%08d.pb' % (job.id))

