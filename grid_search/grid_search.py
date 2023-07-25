#! /usr/bin/env python

import sys
import os
import subprocess
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

global A, b
global seed
global n_rows
global n_cols
global tolerance
global mattype
global nthreads
global x_star
global direct_time
global direct_relative_residual

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-nthreads', type=int, default=8)
    parser.add_argument('-tolerance', type=str, default=1e-6)
    parser.add_argument('-mattype', type=str, default="GA")
    parser.add_argument('-n_rows', type=int, default=1, help="n_rows")
    parser.add_argument('-n_cols', type=int, default=1, help="n_cols")

    args = parser.parse_args()

    return args

def objective(params):

    print (params)

    global A, b
    global seed
    global n_rows
    global n_cols
    global tolerance
    global mattype
    global nthreads
    global x_star
    global direct_time
    global direct_relative_residual

    rls_method = params["rls_method"]
    sketch_operator = params["sketch_operator"]
    sampling_factor = params["sampling_factor"]
    vec_nnz = params["vec_nnz"]
    print ("sampling_factor: ", sampling_factor)

    n,d = A.shape # in the task parameter definition (input matrix), we used term {n, d} instead of {m, n}; the value is stored in the DB file.
    #seed = 1

    grid_search_logfile = "grid_search.db/GRID-SEARCH-n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-mattype_"+str(mattype)+"-tolerance_"+str(tolerance)+".json"
    json_data_arr = []

    if not os.path.exists(grid_search_logfile):
        with open(grid_search_logfile, "w") as f_out:
            f_out.write("[]")

    with open(grid_search_logfile, "r") as f_in:
        json_data_arr = json.load(f_in)

    for func_eval in json_data_arr:
        if func_eval["tuning_parameter"]["rls_method"] == rls_method and\
           func_eval["tuning_parameter"]["sketch_operator"] == sketch_operator and\
           func_eval["tuning_parameter"]["sampling_factor"] == sampling_factor and\
           func_eval["tuning_parameter"]["vec_nnz"] == vec_nnz:
            return

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
    
        x, log = sap(A, b, 0.0, tolerance, d, rng, logging=False, logging_condnum_precond=False)
        parla_time = time.time() - tic
        parla_times.append(parla_time)

        # To reduce experiment time, we do not collect PARLA logs
        x, log = sap(A, b, 0.0, tolerance, d, rng, logging=True, logging_condnum_precond=True)
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

    point = {
        "task_parameter": {
            "n": n,
            "d": d
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
            "error_tolerance": tolerance,
            "dataset": "synthetic_mvt",
            "nthreads": nthreads
        },
        "tuning_parameter": {
            "rls_method": rls_method,
            "sketch_operator": sketch_operator,
            "sampling_factor": float(sampling_factor),
            "vec_nnz": int(vec_nnz)
        },
        "evaluation_result": {
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
        "tuning": "grid_search",
        "parla_logs": parla_logs
    }

    print (point)

    json_data_arr.append(point)
    with open(grid_search_logfile, "w") as f_out:
        json.dump(json_data_arr, f_out, indent=2)

    return np.average(parla_times)

def run_grid_search():

    global A, b
    global seed
    global n_rows
    global n_cols
    global tolerance
    global mattype
    global nthreads
    global x_star
    global direct_time
    global direct_relative_residual

    args = parse_args()
    n_rows = args.n_rows
    print ("n_rows: ", n_rows)
    n_cols = args.n_cols
    print ("n_cols: ", n_cols)
    tolerance = float(args.tolerance)
    print ("tolerance: ", tolerance)
    mattype = str(args.mattype)
    print ("mattype: ", mattype)
    nthreads = args.nthreads
    print ("nthreads: ", nthreads)
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

    for rls_method in ["blendenpik", "lsrn", "newtonsketch"]:
        for sketch_operator in ["sjlt","less_uniform"]:
            for sampling_factor in list(np.arange(1,11,1)):
                for vec_nnz in list(np.arange(1,10,1))+list(np.arange(10,101,10)):
                    params = {
                        "rls_method": rls_method,
                        "sketch_operator": sketch_operator,
                        "sampling_factor": sampling_factor,
                        "vec_nnz": vec_nnz
                    }
                    objective(params)

    print ("Done")

if __name__ == "__main__":

    global A, b
    global seed
    global n_rows
    global n_cols
    global tolerance
    global mattype
    global nthreads
    global x_star
    global direct_time
    global direct_relative_residual

    run_grid_search()

