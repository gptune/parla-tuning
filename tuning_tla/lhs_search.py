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

    parser.add_argument('-nrun', type=int, default=50, help='Number of runs per task')
    #parser.add_argument('-npilot', type=int, default=0, help='Number of initial samples per task')
    parser.add_argument('-mattype', type=str, default="GA")
    parser.add_argument('-nthreads', type=int, default=8)
    parser.add_argument('-n_rows', type=int, default=1, help="n_rows")
    parser.add_argument('-n_cols', type=int, default=1, help="n_cols")
    parser.add_argument('-batch_num', type=int, default=1, help="Batch num")
    parser.add_argument('-failure_handling', type=str, default="highval")
    parser.add_argument('-search_module', type=str, default="LHSMDU")

    args = parser.parse_args()

    return args

def objectives(point):
    global A, b

    global n_rows
    global n_cols
    global mattype
    global nthreads
    global x_star
    global direct_time
    global direct_relative_residual

    global batch_num

    global reference_normalized_residual_errors_to_Axstar

    global failure_handling

    global search_module

    m = point["m"]
    n = point["n"]

    rls_method = point["rls_method"]
    sketch_operator = point["sketch_operator"]
    sampling_factor = point["sampling_factor"]
    vec_nnz = point["vec_nnz"]
    safety_exponent = point["safety_exponent"]
    ref_tolerance = 1e-6
    tolerance = ref_tolerance/(10.0**safety_exponent)

    print ("rls_method: ", rls_method)
    print ("sketch_operator: ", sketch_operator)
    print ("sampling_factor: ", sampling_factor)
    print ("vec_nnz: ", vec_nnz)
    print ("stopping_tolerance: ", tolerance)

    initial_eval = False
    if n_rows == 10000 and n_cols == 1000:
        if rls_method == "blendenpik" and \
           sketch_operator == "sjlt" and \
           sampling_factor == 5.0 and\
           vec_nnz == 50 and\
           safety_exponent == 0:
            initial_eval = True
    elif n_rows == 10000 and n_cols == 2000:
        if rls_method == "blendenpik" and \
           sketch_operator == "sjlt" and \
           sampling_factor == 2.5 and\
           vec_nnz == 50 and\
           safety_exponent == 0:
            initial_eval = True
    else:
        if rls_method == "blendenpik" and \
           sketch_operator == "sjlt" and \
           sampling_factor == 5.0 and\
           vec_nnz == 50 and\
           safety_exponent == 0:
            initial_eval = True

    niter = point["niter"]

    parla_times = []
    parla_logs = []
    relative_residual_errors = []
    residual_errors_to_xstar = []
    residual_errors_to_Axstar = []
    normalized_residual_errors_to_Axstar = []
    adaptive_relative_normal_equation_errors = []
    arnorms = []
    error_bound_checks = []

    niter = 5
    for iter_ in range(niter):
        seed = iter_+1
        rng = np.random.default_rng(seed)
        tic = time.time()

        if rls_method == "blendenpik":
            if sketch_operator == "sjlt":
                sap = rlsq.SPO(oblivious.SkOpSJ(vec_nnz=vec_nnz), sampling_factor=sampling_factor, mode='qr')
            elif sketch_operator == "less_uniform":
                sap = rlsq.SPO(oblivious.SkOpNL(vec_nnz=vec_nnz), sampling_factor=sampling_factor, mode='qr')
            elif sketch_operator == "gaussian":
                sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=sampling_factor, mode='qr')
        elif rls_method == "lsrn":
            if sketch_operator == "sjlt":
                sap = rlsq.SPO(oblivious.SkOpSJ(vec_nnz=vec_nnz), sampling_factor=sampling_factor, mode='svd')
            elif sketch_operator == "less_uniform":
                sap = rlsq.SPO(oblivious.SkOpNL(vec_nnz=vec_nnz), sampling_factor=sampling_factor, mode='svd')
            elif sketch_operator == "gaussian":
                sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=sampling_factor, mode='svd')
        elif rls_method == "newtonsketch":
            if sketch_operator == "sjlt":
                sap = rlsq.SPO(oblivious.SkOpSJ(vec_nnz=vec_nnz), sampling_factor=sampling_factor, mode='svd')
                sap.iterative_solver = PcSS3()
            elif sketch_operator == "less_uniform":
                sap = rlsq.SPO(oblivious.SkOpNL(vec_nnz=vec_nnz), sampling_factor=sampling_factor, mode='svd')
                sap.iterative_solver = PcSS3()
            elif sketch_operator == "gaussian":
                sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=sampling_factor, mode='svd')
                sap.iterative_solver = PcSS3()
    
        x, log = sap(A, b, 0.0, tolerance, n, rng, logging=False, logging_condnum_precond=False)
        parla_time = time.time() - tic
        parla_times.append(parla_time)

        # To reduce experiment time, we do not collect PARLA logs
        x, log = sap(A, b, 0.0, tolerance, n, rng, logging=True, logging_condnum_precond=True)
        parla_logs.append({
            "time_sketch": log.time_sketch,
            "time_factor": log.time_factor,
            "time_presolve": log.time_presolve,
            "time_convert": log.time_convert,
            "time_setup": log._time_setup,
            "time_iterate": log.time_iterate,
            "times": list(log.times),
            "errors": list(log.errors),
            "error_desc": log.error_desc,
            "condnum_precond" : log.condnum_precond
            })

        r = np.matmul(A, x) - b
        relative_residual_error = norm(r)/norm(b)
        relative_residual_errors.append(relative_residual_error)
        residual_errors_to_xstar.append(norm(x - x_star))
        residual_errors_to_Axstar.append(norm(A@x - A@x_star))
        normalized_residual_errors_to_Axstar.append(norm(A@x - A@x_star)/norm(A@x-b))
        adaptive_relative_normal_equation_error = norm((A.T@A)@x-(A.T@b))/(norm(A)*norm(A@x-b))
        adaptive_relative_normal_equation_errors.append(adaptive_relative_normal_equation_error)
        arnorms.append(norm(A.T@(b-A@x)))
        error_bound_checks.append(norm(A@(x-x_star)) / norm(A@x_star))

    if initial_eval == True:
        reference_normalized_residual_errors_to_Axstar = np.average(normalized_residual_errors_to_Axstar)
        ret = np.average(parla_times)
    else:
        if failure_handling == "highval":
            if np.average(normalized_residual_errors_to_Axstar) > 10*(reference_normalized_residual_errors_to_Axstar):
                ret = 2 * np.average(parla_times)
            else:
                ret = np.average(parla_times)
        elif failure_handling == "none":
            ret = np.average(parla_times)

    ret_additional_dict = {
        "parla_times": parla_times,
        "relative_residual_errors": relative_residual_errors,
        "adaptive_relative_normal_equation_errors": adaptive_relative_normal_equation_errors,
        "arnorms": arnorms,
        "residual_errors_to_Axstar": residual_errors_to_Axstar,
        "normalized_residual_errors_to_Axstar": normalized_residual_errors_to_Axstar,
        "error_bound_checks": error_bound_checks
    }

    return [ret], ret_additional_dict

def main():

    global A, b

    global seed
    global n_rows
    global n_cols
    global mattype
    global nthreads
    global x_star
    global direct_time
    global direct_relative_residual

    global batch_num

    global reference_normalized_residual_errors_to_Axstar

    global failure_handling

    global search_module

    args = parse_args()
    n_rows = args.n_rows
    print ("n_rows: ", n_rows)
    n_cols = args.n_cols
    print ("n_cols: ", n_cols)
    mattype = str(args.mattype)
    print ("mattype: ", mattype)
    nthreads = args.nthreads
    print ("nthreads: ", nthreads)
    nrun = args.nrun
    print ("nrun: ", nrun)
    npilot = nrun
    print ("npilot: ", npilot)
    batch_num = args.batch_num
    print ("batch_num: ", batch_num)
    failure_handling = args.failure_handling
    print ("failure_handling: ", failure_handling)
    search_module = args.search_module
    print ("search_module: ", search_module)

    niter = 5

    A = np.genfromtxt("../input/synthetic_mvt/data-nrows_"+str(n_rows)+"-ncols_"+str(n_cols)+"-mattype_"+str(mattype)+".csv", delimiter=',', skip_header=1, dtype=np.float64)
    A = np.delete(A, 0, 1) # the first row of the synthetic input data is meta information, so we remove that here.

    b = np.genfromtxt("../input/synthetic_mvt/result-nrows_"+str(n_rows)+"-ncols_"+str(n_cols)+"-mattype_"+str(mattype)+".csv", delimiter=',', skip_header=1, dtype=np.float64)
    b = np.delete(b, 0, 1) # the first row of the synthetic input data is meta information, so we remove that here.
    b = b.ravel()

    direct_tic = time.time()
    x_star = np.linalg.lstsq(A, b, rcond=None)[0]
    direct_time = time.time() - direct_tic
    direct_relative_residual = norm(np.matmul(A, x_star) - b) / norm(b)
    print("Direct Time", direct_time)
    print("Direct Relative Residual", np.linalg.norm(A.dot(x_star)-b) / np.linalg.norm(b))

    n_rows, n_cols = A.shape
    print ("m: ", n_rows)
    print ("n: ", n_cols)

    """ tuning meta information """
    if search_module == "LHSMDU":
        tuning_metadata = {
            "tuning_problem_name": "LHSMDU-SEARCH-failure_handling_"+str(failure_handling)+"-n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-mattype_"+str(mattype)+"-batch_num_"+str(batch_num),
            "historydb_path": "./lhsmdu.db/",
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
            "no_load_check": "yes"
        }
    elif search_module == "LHSOPENTURNS":
        tuning_metadata = {
            "tuning_problem_name": "LHSOPENTURNS-SEARCH-failure_handling_"+str(failure_handling)+"-n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-mattype_"+str(mattype)+"-batch_num_"+str(batch_num),
            "historydb_path": "./lhsopenturns.db/",
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
            "no_load_check": "yes"
        }
    else:
        tuning_metadata = {
            "tuning_problem_name": "LHSMDU-SEARCH-failure_handling_"+str(failure_handling)+"-n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-mattype_"+str(mattype)+"-batch_num_"+str(batch_num),
            "historydb_path": "./lhsmdu.db/",
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
            "no_load_check": "yes"
        }

    (machine, processor, nodes, cores) = GetMachineConfiguration(meta_dict = tuning_metadata)
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

    """ input space """
    m = Integer(1000, 100000, transform="normalize", name="m")
    n = Integer(1000, 10000, transform="normalize", name="n")
    input_space = Space([m,n])

    """ tuning parameter space """
    rls_method = Categoricalnorm (["blendenpik", "lsrn", "newtonsketch"], transform="onehot", name="rls_method")
    sketch_operator = Categoricalnorm (["sjlt", "less_uniform"], transform="onehot", name="sketch_operator")
    if n_rows == 10000 and n_cols == 1000:
        sampling_factor = Real(1.0, 10.0, transform="normalize", name="sampling_factor")
    elif n_rows == 10000 and n_cols == 2000:
        sampling_factor = Real(1.0, 5.0, transform="normalize", name="sampling_factor")
    else:
        sampling_factor = Real(1.0, 10.0, transform="normalize", name="sampling_factor")
    vec_nnz = Integer(1, 100, transform="normalize", name="vec_nnz")
    safety_exponent = Integer(0, 4, transform="normalize", name="safety_exponent")
    parameter_space = Space([rls_method,sketch_operator,sampling_factor,vec_nnz,safety_exponent])

    """ ouput space """
    obj = Real(float("-Inf"), float("Inf"), name="obj", optimize=True)
    output_space = Space([obj])

    """ constant variables """
    constants = {
        "niter": niter,
        "dataset": "synthetic",
        "nthreads": nthreads
    }

    """ constraints """
    constraints = {}

    """ gptune setting """
    problem = TuningProblem(input_space, parameter_space, output_space, objectives, constraints, constants=constants)
    historydb = HistoryDB(meta_dict=tuning_metadata)
    computer = Computer(nodes=nodes, cores=cores, hosts=None)
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

    if search_module == "LHSMDU":
        options['sample_class'] = 'SampleLHSMDU'
        options['sample_random_seed'] = batch_num
    elif search_module == "LHSOPENTURNS":
        options['sample_class'] = 'SampleOpenTURNS'
        options['sample_random_seed'] = batch_num
    else:
        options['sample_class'] = 'SampleLHSMDU'
        options['sample_random_seed'] = batch_num
    options['model_class'] = 'Model_GPy_LCM'
    options['model_random_seed'] = batch_num
    options['search_class'] = 'SearchPyGMO'
    options['search_random_seed'] = batch_num
    options.validate(computer=computer)

    """ run gptune """
    giventask = [[n_rows, n_cols]]
    NI=len(giventask)
    NS=nrun

    gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
    if n_rows == 10000 and n_cols == 1000:
        gt.EvaluateObjective(T=[n_rows, n_cols], P=[["blendenpik", "sjlt", 5.0, 50, 0]])
    elif n_rows == 10000 and n_cols == 2000:
        gt.EvaluateObjective(T=[n_rows, n_cols], P=[["blendenpik", "sjlt", 2.5, 50, 0]])
    else:
        gt.EvaluateObjective(T=[n_rows, n_cols], P=[["blendenpik", "sjlt", 5.0, 50, 0]])
    (data, modeler, stats) = gt.MLA(NS=NS, NS1=npilot, NI=NI, Tgiven=giventask)
    print("stats: ", stats)
    #(data, modeler, stats) = gt.SLA(NS=NS, NS1=npilot, Tgiven=giventask)
    #print("stats: ", stats)

    """ Print all input and parameter samples """
    print("    T: ", (data.I))
    print("    Ps ", data.P)
    print("    Os ", data.O.tolist())
    print('    Popt ', data.P[np.argmin(data.O)], 'Oopt ', min(data.O)[0], 'nth ', np.argmin(data.O))

if __name__ == "__main__":

    main()
