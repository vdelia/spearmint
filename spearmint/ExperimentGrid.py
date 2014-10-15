##
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
import os
import sys
import tempfile
import cPickle

import numpy        as np
import numpy.random as npr

from sobol_lib     import *
from helpers       import *

import logging

CANDIDATE_STATE = 0
SUBMITTED_STATE = 1
RUNNING_STATE   = 2
COMPLETE_STATE  = 3
BROKEN_STATE    = -1

EXPERIMENT_GRID_FILE = 'expt-grid.pkl'

class ExperimentGrid:

    @staticmethod
    def job_running(expt_dir, id):
        expt_grid = ExperimentGrid(expt_dir)
        expt_grid.set_running(id)

    @staticmethod
    def job_complete(expt_dir, id, value, duration):
        logging.info("setting job %d complete", id)
        expt_grid = ExperimentGrid(expt_dir)
        expt_grid.set_complete(id, value, duration)
        logging.info("set...")

    @staticmethod
    def job_broken(expt_dir, id):
        expt_grid = ExperimentGrid(expt_dir)
        expt_grid.set_broken(id)

    def __init__(self, expt_dir, variables=None, grid_size=None, grid_seed=1):
        self.expt_dir = expt_dir
        self.jobs_pkl = os.path.join(expt_dir, EXPERIMENT_GRID_FILE)


        # Set up the grid for the first time if it doesn't exist.
        if variables is not None and not os.path.exists(self.jobs_pkl):
            self.seed     = grid_seed
            self.vmap     = GridMap(variables, grid_size)
            self.grid     = self._hypercube_grid(self.vmap.card(), grid_size)
            self.status   = np.zeros(grid_size, dtype=int) + CANDIDATE_STATE
            self.values   = np.zeros(grid_size) + np.nan
            self.durs     = np.zeros(grid_size) + np.nan
            self.proc_ids = np.zeros(grid_size, dtype=int)
            self._save_jobs()

        # Or load in the grid from the pickled file.
        else:
            self._load_jobs()


    def __del__(self):
        self._save_jobs()

    def get_grid(self):
        return self.grid, self.values, self.durs

    def get_candidates(self):
        return np.nonzero(self.status == CANDIDATE_STATE)[0]

    def get_pending(self):
        return np.nonzero((self.status == SUBMITTED_STATE) | (self.status == RUNNING_STATE))[0]

    def get_complete(self):
        return np.nonzero(self.status == COMPLETE_STATE)[0]

    def get_broken(self):
        return np.nonzero(self.status == BROKEN_STATE)[0]

    def get_params(self, index):
        return self.vmap.get_params(self.grid[index,:])

    def get_best(self):
        finite = self.values[np.isfinite(self.values)]
        if len(finite) > 0:
            cur_min = np.min(finite)
            index   = np.nonzero(self.values==cur_min)[0][0]
            return cur_min, index
        else:
            return np.nan, -1

    def get_proc_id(self, id):
        return self.proc_ids[id]

    def add_to_grid(self, candidate):
        # Checks to prevent numerical over/underflow from corrupting the grid
        candidate[candidate > 1.0] = 1.0
        candidate[candidate < 0.0] = 0.0

        # Set up the grid
        self.grid   = np.vstack((self.grid, candidate))
        self.status = np.append(self.status, np.zeros(1, dtype=int) +
                                int(CANDIDATE_STATE))

        self.values = np.append(self.values, np.zeros(1)+np.nan)
        self.durs   = np.append(self.durs, np.zeros(1)+np.nan)
        self.proc_ids = np.append(self.proc_ids, np.zeros(1,dtype=int))

        # Save this out.
        self._save_jobs()
        return self.grid.shape[0]-1

    def set_candidate(self, id):
        self.status[id] = CANDIDATE_STATE
        self._save_jobs()

    def set_submitted(self, id, proc_id):
        self.status[id] = SUBMITTED_STATE
        self.proc_ids[id] = proc_id
        self._save_jobs()

    def set_running(self, id):
        self.status[id] = RUNNING_STATE
        self._save_jobs()

    def set_complete(self, id, value, duration):
        self.status[id] = COMPLETE_STATE
        self.values[id] = value
        self.durs[id]   = duration
        self._save_jobs()

    def set_broken(self, id):
        self.status[id] = BROKEN_STATE
        self._save_jobs()

    def _load_jobs(self):
        fh   = open(self.jobs_pkl, 'r')
        jobs = cPickle.load(fh)
        fh.close()

        self.vmap   = jobs['vmap']
        self.grid   = jobs['grid']
        self.status = jobs['status']
        self.values = jobs['values']
        self.durs   = jobs['durs']
        self.proc_ids = jobs['proc_ids']

    def _save_jobs(self):

        # Write everything to a temporary file first.
        fh = tempfile.NamedTemporaryFile(mode='w', delete=False)
        cPickle.dump({ 'vmap'   : self.vmap,
                       'grid'   : self.grid,
                       'status' : self.status,
                       'values' : self.values,
                       'durs'   : self.durs,
                       'proc_ids' : self.proc_ids }, fh, protocol=-1)
        fh.close()

        # Use an atomic move for better NFS happiness.
        cmd = 'mv "%s" "%s"' % (fh.name, self.jobs_pkl)
        os.system(cmd) # TODO: Should check system-dependent return status.

    def _hypercube_grid(self, dims, size):
        # Generate from a sobol sequence
        sobol_grid = np.transpose(i4_sobol_generate(dims,size,self.seed))

        return sobol_grid

class GridMap():

    valid_types = set(["int", "float", "enum"])
    min_max_validators = {"int" : int, "float": float}

    def accept_variable(self, variable):
        validator = GridMap.min_max_validators.get(variable["type"], None)
        assert variable["type"] in GridMap.valid_types
        
        new_variable = dict(variable)
            
        if validator is not None:
            new_variable["min"] = validator(new_variable["min"])
            new_variable["max"] = validator(new_variable["max"])
        return new_variable

    def __init__(self, variables, grid_size):
        self.variables   = []
        self.cardinality = 0

        # Count the total number of dimensions and roll into new format.
        for variable in variables:
            self.cardinality += variable["size"]
            variable = self.accept_variable(variable)
            self.variables.append(variable)

        logging.info("Optimizing over %d dimensions", self.cardinality)

    def get_params(self, u):
        if u.shape[0] != self.cardinality:
            raise Exception("Hypercube dimensionality is incorrect.")

        params = {}
        index  = 0
        for variable in self.variables:
            name = variable['name']
            values = None

            if variable['type'] == 'int':
                v = []
                for dd in xrange(variable['size']):
                    v.append(variable['min'] + self._index_map(u[index], variable['max']-variable['min']+1))
                    index += 1
                values = np.array(v, dtype=int)

            elif variable['type'] == 'float':
                v = []
                for dd in xrange(variable['size']):
                    v.append(variable['min'] + u[index]*(variable['max']-variable['min']))
                    index += 1
                values = np.array(v)

            elif variable['type'] == 'enum':
                v = []
                for dd in xrange(variable['size']):
                    ii = self._index_map(u[index], len(variable['options']))
                    index += 1
                    v.append(variable['options'][ii])
                values = v

            else:
                raise Exception("Unknown parameter type.")

            assert name not in params, "multiple definitions of %s" % name
            params[name] = values

        return params

    def card(self):
        return self.cardinality

    def _index_map(self, u, items):
        u = np.max((u, 0.0))
        u = np.min((u, 1.0))
        return int(np.floor((1-np.finfo(float).eps) * u * float(items)))
