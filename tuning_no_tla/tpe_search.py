#! /usr/bin/env python

import sys
import os
import subprocess
sys.path.insert(0, os.path.abspath(__file__ + "/../../GPTune/GPTune/"))

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

from hyperopt import tpe, hp, fmin
from hyperopt.fmin import generate_trials_to_calculate

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-nrun', type=int, default=50, help='Number of runs per task')
    parser.add_argument('-npilot', type=int, default=0, help='Number of initial samples per task')
    parser.add_argument('-mattype', type=str, default="GA")
    parser.add_argument('-nthreads', type=int, default=8)
    parser.add_argument('-n_rows', type=int, default=1, help="n_rows")
    parser.add_argument('-n_cols', type=int, default=1, help="n_cols")
    parser.add_argument('-batch_num', type=int, default=1, help="Batch num")
    parser.add_argument('-failure_handling', type=str, default="highval")

    args = parser.parse_args()

    return args

def objective(params):

    global A, b

    global n_rows
    global n_cols
    global mattype
    global nthreads
    global x_star
    global direct_time
    global direct_relative_residual

    global npilot
    global batch_num

    global reference_normalized_residual_errors_to_Axstar

    global failure_handling

    print ("Objective")

    m , n = A.shape

    rls_method = params["rls_method"]
    print ("rls_method: ", rls_method)
    if rls_method == 1:
        rls_method = "blendenpik"
    elif rls_method == 2:
        rls_method = "lsrn"
    elif rls_method == 3:
        rls_method = "newtonsketch"
    sketch_operator = params["sketch_operator"]
    print ("sketch_operator: ", sketch_operator)
    if sketch_operator == 1:
        sketch_operator = "sjlt"
    elif sketch_operator == 2:
        sketch_operator = "less_uniform"
    sampling_factor = params["sampling_factor"]
    vec_nnz = int(params["vec_nnz"])
    safety_exponent = int(params["safety_exponent"])
    ref_tolerance = 1e-6
    tolerance = ref_tolerance/(10.0**safety_exponent)
    print ("rls_method: ", rls_method)
    print ("sketch_operator: ", sketch_operator)
    print ("sampling_factor: ", sampling_factor)
    print ("vec_nnz: ", vec_nnz)

    tpe_search_logfile = "tpe.db/TPE-SEARCH-failure_handling_"+str(failure_handling)+"-n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-mattype_"+str(mattype)+"-npilot_"+str(npilot)+"-batch_num_"+str(batch_num)+".json"
    json_data_arr = []

    if not os.path.exists(tpe_search_logfile):
        with open(tpe_search_logfile, "w") as f_out:
            f_out.write("[]")

    if os.path.exists(tpe_search_logfile):
        with open(tpe_search_logfile, "r") as f_in:
            json_data_arr = json.load(f_in)

    initial_eval = False
    if rls_method == "blendenpik" and \
       sketch_operator == "sjlt" and \
       sampling_factor == 5.0 and\
       vec_nnz == 50 and\
       safety_exponent == 0:
        initial_eval = True

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

    point = {
        "task_parameter": {
            "m": m,
            "n": n
        },
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
        "constants": {
            "niter": niter,
            "dataset": "synthetic",
            "nthreads": nthreads
        },
        "tuning_parameter": {
            "rls_method": rls_method,
            "sketch_operator": sketch_operator,
            "sampling_factor": float(sampling_factor),
            "vec_nnz": int(vec_nnz),
            "safety_exponent": int(safety_exponent)
        },
        "evaluation_result": {
            "obj": ret,
            "wall_clock_time": np.average(parla_times),
            "relative_residual_error": np.average(relative_residual_errors),
            "adaptive_relative_normal_equation_error": np.average(adaptive_relative_normal_equation_errors),
            "arnorm": np.average(arnorms),
            "residual_error_to_Axstar": np.average(residual_errors_to_Axstar),
            "normalized_residual_error_to_Axstar": np.average(normalized_residual_errors_to_Axstar),
            "error_bound_check": np.average(error_bound_checks)
        },
        "evaluation_detail": {
            "wall_clock_time": {
                "evaluations": parla_times,
                "objective_scheme": "average"
            },
            "relative_residual_error": {
                "evaluations": relative_residual_errors,
                "objective_scheme": "average"
            },
            "adaptive_relative_normal_equation_errors": {
                "evaluations": adaptive_relative_normal_equation_errors,
                "objective_scheme": "average"
            },
            "arnorms": {
                "evaluations": arnorms,
                "objective_scheme": "average"
            },
            "error_bound_checks": {
                "evaluations": error_bound_checks,
                "objective_scheme": "average"
            },
            "residual_errors_to_Axstar": {
                "evaluations": residual_errors_to_Axstar,
                "objective_scheme": "average"
            },
            "normalized_residual_errors_to_Axstar": {
                "evaluations": normalized_residual_errors_to_Axstar,
                "objective_scheme": "average"
            }
        },
        "source": "measure",
        "tuning": "tpe",
        "parla_logs": parla_logs
    }
    print (point)

    json_data_arr.append(point)
    with open(tpe_search_logfile, "w") as f_out:
        json.dump(json_data_arr, f_out, indent=2)

    return ret

def run_tpe_search():

    global A, b

    global n_rows
    global n_cols
    global mattype
    global nthreads
    global x_star
    global direct_time
    global direct_relative_residual

    global npilot
    global batch_num

    global reference_normalized_residual_errors_to_Axstar

    global failure_handling

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
    npilot = args.npilot
    print ("npilot: ", npilot)
    batch_num = args.batch_num
    print ("batch_num: ", batch_num)
    failure_handling = args.failure_handling
    print ("failure_handling: ", failure_handling)

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

    # Define the search space of x between -10 and 10.
    #space = hp.uniform('x', -10, 10)
    #space = {
    #    'rls_method': hp.choice('rls_method', ['blendenpik','lsrn','newtonsketch']),
    #    'sketch_operator': hp.choice('sketch_operator', ['sjlt','less_uniform']),
    #    'sampling_factor': hp.uniform('sampling_factor', 1.0, 10),
    #    'vec_nnz': hp.randint('vec_nnz', 1, 100)
    #}

    space = {
        'rls_method': hp.randint('rls_method', 1, 4),
        'sketch_operator': hp.randint('sketch_operator', 1, 3),
        'sampling_factor': hp.uniform('sampling_factor', 1.0, 10),
        'vec_nnz': hp.randint('vec_nnz', 1, 101),
        'safety_exponent': hp.randint('safety_exponent', 0, 5),
    }


    #best = fmin(
    #    fn=objective, # Objective Function to optimize
    #    space=space, # Hyperparameter's Search Space
    #    algo=tpe.suggest, # Optimization algorithm
    #    max_evals=nrun # Number of optimization attempts
    #    )

    #trials = generate_trials_to_calculate([{'rls_method':'blendenpik', 'sketch_operator':'sjlt', 'sampling_factor':5.0, 'vec_nnz':50 }])
    if npilot == 0:
        trials = generate_trials_to_calculate([{'rls_method':1, 'sketch_operator':1, 'sampling_factor':5.0, 'vec_nnz':50, 'safety_exponent':0 }])
    elif npilot == 10:
        with open("lhsmdu.sample/LHSMDU-SAMPLE-n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-n_samples_"+str(npilot)+"-batch_num_"+str(batch_num)+".json") as f_in:
            samples = json.load(f_in)
            P = []
            for sample in samples:
                if sample["rls_method"] == "blendenpik":
                    if sample["sketch_operator"] == "sjlt":
                        P_ = { "rls_method": 1,
                               "sketch_operator": 1,
                               "sampling_factor": sample["sampling_factor"],
                               "vec_nnz": sample["vec_nnz"],
                               "safety_exponent": sample["safety_exponent"] }
                    elif sample["sketch_operator"] == "less_uniform":
                        P_ = { "rls_method": 1,
                               "sketch_operator": 2,
                               "sampling_factor": sample["sampling_factor"],
                               "vec_nnz": sample["vec_nnz"],
                               "safety_exponent": sample["safety_exponent"] }
                elif sample["rls_method"] == "lsrn":
                    if sample["sketch_operator"] == "sjlt":
                        P_ = { "rls_method": 2,
                               "sketch_operator": 1,
                               "sampling_factor": sample["sampling_factor"],
                               "vec_nnz": sample["vec_nnz"],
                               "safety_exponent": sample["safety_exponent"] }
                    elif sample["sketch_operator"] == "less_uniform":
                        P_ = { "rls_method": 2,
                               "sketch_operator": 2,
                               "sampling_factor": sample["sampling_factor"],
                               "vec_nnz": sample["vec_nnz"],
                               "safety_exponent": sample["safety_exponent"] }
                elif sample["rls_method"] == "newtonsketch":
                    if sample["sketch_operator"] == "sjlt":
                        P_ = { "rls_method": 3,
                               "sketch_operator": 1,
                               "sampling_factor": sample["sampling_factor"],
                               "vec_nnz": sample["vec_nnz"],
                               "safety_exponent": sample["safety_exponent"] }
                    elif sample["sketch_operator"] == "less_uniform":
                        P_ = { "rls_method": 3,
                               "sketch_operator": 2,
                               "sampling_factor": sample["sampling_factor"],
                               "vec_nnz": sample["vec_nnz"],
                               "safety_exponent": sample["safety_exponent"] }
                P.append(P_)
            trials = generate_trials_to_calculate(P)
    else:
        trials = generate_trials_to_calculate([{'rls_method':1, 'sketch_operator':1, 'sampling_factor':5.0, 'vec_nnz':50, 'safety_exponent': 0 }])

    best = fmin(
        fn=objective, # Objective Function to optimize
        space=space, # Hyperparameter's Search Space
        trials=trials, # Provide evaluation of the initial parameter configuration
        algo=tpe.suggest, # Optimization algorithm
        max_evals=nrun # Number of optimization attempts
        )
    print(best)

    return

if __name__ == "__main__":

    run_tpe_search()

