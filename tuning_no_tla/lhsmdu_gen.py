#! /usr/bin/env python

import sys
import os
import subprocess
sys.path.insert(0, os.path.abspath(__file__ + "/../../GPTune/GPTune/"))

from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import * # import all
import argparse
import json

import time
import math
import numpy as np
from numpy.linalg import norm
import scipy.linalg as la
import parla as rla
import parla.drivers.least_squares as rlsq
import parla.utils.sketching as usk
import parla.comps.sketchers.oblivious as oblivious
import parla.utils.stats as ustats
import parla.tests.matmakers as matmakers
from parla.comps.determiter.saddle import PcSS3

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-n_rows', type=int, default=50000, help="n_rows")
    parser.add_argument('-n_cols', type=int, default=1000, help="n_cols")
    parser.add_argument('-n_samples', type=int, default=10, help="n_samples")
    parser.add_argument('-batch_num', type=int, default=1, help="Batch num")

    args = parser.parse_args()

    return args

def objectives(point):

    return [-1]

def main():

    args = parse_args()
    n_samples = args.n_samples
    batch_num = args.batch_num
    n_rows = args.n_rows
    n_cols = args.n_cols

    """ tuning meta information """
    tuning_metadata = {
        "tuning_problem_name": "LHSMDU-SAMPLE-n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-n_samples_"+str(n_samples)+"-batch_num_"+str(batch_num)+"-temp",
        "machine_configuration": {
            "machine_name": "millennium",
            "xeon": { "nodes": 1, "cores": 8 }
        },
        "software_configuration": {
            "parla": [0,1,5],
            "numpy": [1,21,2],
            "scipy": [1,7,3],
            "mkl": [2022,0,0]
        },
        "historydb_path": "./lhsmdu.sample/",
        "no_load_check": "yes"
    }

    """ input space """
    m = Integer(1000, 100000, transform="normalize", name="m")
    n = Integer(1000, 10000, transform="normalize", name="n")
    input_space = Space([m,n])

    """ tuning parameter space """
    rls_method = Categoricalnorm (["blendenpik", "lsrn", "newtonsketch"], transform="onehot", name="rls_method")
    sketch_operator = Categoricalnorm (["sjlt", "less_uniform"], transform="onehot", name="sketch_operator")
    sampling_factor = Real(1.0, 10.0, transform="normalize", name="sampling_factor")
    vec_nnz = Integer(1, 100, transform="normalize", name="vec_nnz")
    safety_exponent = Integer(0, 4, transform="normalize", name="safety_exponent")
    parameter_space = Space([rls_method,sketch_operator,sampling_factor,vec_nnz,safety_exponent])

    """ ouput space """
    obj = Real(float("-Inf"), float("Inf"), name="obj", optimize=True)
    output_space = Space([obj])

    """ constant variables """
    constants = {}

    """ constraints """
    constraints = {}

    """ GPTune setting """
    problem = TuningProblem(input_space, parameter_space, output_space, objectives, constraints, constants=constants)
    historydb = HistoryDB(meta_dict=tuning_metadata)
    computer = Computer(nodes=1, cores=8, hosts=None)
    data = Data(problem)

    options = Options()
    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False
    options['objective_evaluation_parallelism'] = False
    options['objective_multisample_threads'] = 1
    options['objective_multisample_processes'] = 1
    options['model_processes'] = 1
    options['model_restarts'] = 1
    options['model_kern'] = 'RBF'

    options['sample_class'] = 'SampleLHSMDU'
    options['sample_random_seed'] = batch_num
    options['model_class'] = 'Model_GPy_LCM'
    options['model_random_seed'] = batch_num
    options['search_class'] = 'SearchPyMoo'
    options['search_random_seed'] = batch_num
    options.validate(computer=computer)

    """ run GPTune """
    giventask = [n_rows, n_cols]

    gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
    gt.EvaluateObjective(T=[n_rows, n_cols], P=[["blendenpik", "sjlt", 5.0, 50, 0]])
    (data, modeler, stats) = gt.SLA(NS=n_samples, NS1=n_samples, Tgiven=giventask)
    print("stats: ", stats)

    with open ("lhsmdu.sample/LHSMDU-SAMPLE-n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-n_samples_"+str(n_samples)+"-batch_num_"+str(batch_num)+"-temp.json", "r") as f_in:
        function_evaluations_ = json.load(f_in)["func_eval"]
    with open ("lhsmdu.sample/LHSMDU-SAMPLE-n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-n_samples_"+str(n_samples)+"-batch_num_"+str(batch_num)+".json", "w") as f_out:
        samples = []
        for func_eval_ in function_evaluations_:
            sample = func_eval_["tuning_parameter"]
            samples.append(sample)
        json.dump(samples, f_out, indent=2)

if __name__ == "__main__":

    main()
