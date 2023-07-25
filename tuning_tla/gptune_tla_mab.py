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

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-nrun', type=int, default=50, help='Number of runs per task')
    parser.add_argument('-npilot', type=int, default=0, help='Number of initial samples per task')
    parser.add_argument('-nthreads', type=int, default=8)

    parser.add_argument('-mattype', type=str, default="GA")
    parser.add_argument('-n_rows', type=int, default=1, help="n_rows")
    parser.add_argument('-n_cols', type=int, default=1, help="n_cols")
    parser.add_argument('-source_mattype', type=str, default="GA")
    parser.add_argument('-source_n_rows', type=int, default=1, help="n_rows")
    parser.add_argument('-source_n_cols', type=int, default=1, help="n_cols")
    parser.add_argument('-failure_handling', type=str, default="highval")
    parser.add_argument('-source_failure_handling', type=str, default="highval")
    parser.add_argument('-mab_policy', type=str, default="source_greedy")

    parser.add_argument('-batch_num', type=int, default=1, help="Batch num")

    args = parser.parse_args()

    return args

def objectives(point):

    import parla as rla
    import parla.drivers.least_squares as rlsq
    import parla.utils.sketching as usk
    import parla.comps.sketchers.oblivious as oblivious
    import parla.utils.stats as ustats
    import parla.tests.matmakers as matmakers
    from parla.comps.determiter.saddle import PcSS3

    global A, b

    global mattype
    global n_rows
    global n_cols
    global source_mattype
    global source_n_rows
    global source_n_cols

    global nthreads
    global x_star
    global direct_time
    global direct_relative_residual

    global batch_num

    global reference_normalized_residual_error_to_Axstar

    global failure_handling
    global source_failure_handling
    global mab_policy

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

    initial_eval = False
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
        reference_normalized_residual_error_to_Axstar = np.average(normalized_residual_errors_to_Axstar)
        ret = np.average(parla_times)
    else:
        if failure_handling == "highval":
            if np.average(normalized_residual_errors_to_Axstar) > 10*(reference_normalized_residual_error_to_Axstar):
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

def LoadSourceFunctionEvaluations(rls_method, sketch_operator):

    global A, b

    global seed

    global mattype
    global n_rows
    global n_cols

    global source_mattype
    global source_n_rows
    global source_n_cols

    global nthreads
    global x_star
    global direct_time
    global direct_relative_residual

    global batch_num

    global reference_normalized_residual_error_to_Axstar

    global failure_handling
    global source_failure_handling
    global mab_policy

    with open("lhsmdu.db/LHSMDU-SEARCH-failure_handling_"+str(source_failure_handling)+"-n_rows_"+str(source_n_rows)+"-n_cols_"+str(source_n_cols)+"-mattype_"+str(source_mattype)+"-batch_num_1.json") as f_in:
        function_evaluations = []
        function_evaluations_ = json.load(f_in)["func_eval"][0:100]
        #print ("loaded function evaluations: ", function_evaluations)
        for func_eval_ in function_evaluations_:
            func_eval = copy.deepcopy(func_eval_)

            if func_eval_["tuning_parameter"]["rls_method"] == rls_method and\
               func_eval_["tuning_parameter"]["sketch_operator"] == sketch_operator:
                   func_eval["tuning_parameter"].pop("rls_method")
                   func_eval["tuning_parameter"].pop("sketch_operator")

                   func_eval["constants"]["rls_method"] = rls_method
                   func_eval["constants"]["sketch_operator"] = sketch_operator

                   function_evaluations.append(func_eval)

        best_obj = None
        best_func_eval = None
        reference_normalized_residual_error_to_Axstar = None

        for i in range(len(function_evaluations)):
            func_eval = function_evaluations[i]
            wall_clock_time = np.average(func_eval["additional_output"]["parla_times"])
            normalized_residual_error_to_Axstar = np.average(func_eval["additional_output"]["normalized_residual_errors_to_Axstar"])
            if i == 0:
                reference_normalized_residual_error_to_Axstar = normalized_residual_error_to_Axstar

            if i == 0:
                best_obj = wall_clock_time
                best_func_eval = func_eval
            else:
                if failure_handling == "highval":
                    if np.average(normalized_residual_error_to_Axstar) > 10*(reference_normalized_residual_error_to_Axstar):
                        pass
                    else:
                        if wall_clock_time < best_obj:
                            best_obj = wall_clock_time
                            best_func_eval = func_eval
                elif failure_handling == "none":
                    if wall_clock_time < best_obj:
                        best_obj = wall_clock_time
                        best_func_eval = func_eval

        return [function_evaluations], best_func_eval

def LoadBestSourceFunctionEvaluation():

    global A, b

    global seed

    global mattype
    global n_rows
    global n_cols

    global source_mattype
    global source_n_rows
    global source_n_cols

    global nthreads
    global x_star
    global direct_time
    global direct_relative_residual

    global batch_num

    global reference_normalized_residual_error_to_Axstar

    global failure_handling
    global source_failure_handling
    global mab_policy

    with open("lhsmdu.db/LHSMDU-SEARCH-failure_handling_"+str(source_failure_handling)+"-n_rows_"+str(source_n_rows)+"-n_cols_"+str(source_n_cols)+"-mattype_"+str(source_mattype)+"-batch_num_1.json") as f_in:
        function_evaluations = json.load(f_in)["func_eval"][0:100]
        #print ("loaded function evaluations: ", function_evaluations)

        best_obj = None
        best_func_eval = None
        reference_normalized_residual_error_to_Axstar = None

        for i in range(len(function_evaluations)):
            func_eval = function_evaluations[i]
            wall_clock_time = np.average(func_eval["additional_output"]["parla_times"])
            normalized_residual_error_to_Axstar = np.average(func_eval["additional_output"]["normalized_residual_errors_to_Axstar"])
            if i == 0:
                reference_normalized_residual_error_to_Axstar = normalized_residual_error_to_Axstar

            if i == 0:
                best_obj = wall_clock_time
                best_func_eval = func_eval
            else:
                if failure_handling == "highval":
                    if np.average(normalized_residual_error_to_Axstar) > 10*(reference_normalized_residual_error_to_Axstar):
                        pass
                    else:
                        if wall_clock_time < best_obj:
                            best_obj = wall_clock_time
                            best_func_eval = func_eval
                elif failure_handling == "none":
                    if wall_clock_time < best_obj:
                        best_obj = wall_clock_time
                        best_func_eval = func_eval

        return best_func_eval

    return None

def RunTLA_SourceGreedy(nrun):

    global A, b

    global seed

    global mattype
    global n_rows
    global n_cols

    global source_mattype
    global source_n_rows
    global source_n_cols

    global nthreads
    global x_star
    global direct_time
    global direct_relative_residual

    global batch_num

    global reference_normalized_residual_error_to_Axstar

    global failure_handling
    global source_failure_handling
    global mab_policy

    if os.path.exists("gptune_tla.db/GPTUNE-TLA_SourceGreedy-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num)+".json"):
        print ("experiment was already performed")
        print ("gptune_tla.db/GPTUNE-TLA_SourceGreedy-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num)+".json")
        return

    for NS in range(1, nrun+1, 1):
        """ tuning meta information """
        tuning_metadata = {
            "tuning_problem_name": "GPTUNE-TLA_SourceGreedy-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num),
            "historydb_path": "gptune_tla.db",
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
    
        niter = 5
    
        """ input space """
        m = Integer(1000, 100000, transform="normalize", name="m")
        n = Integer(1000, 10000, transform="normalize", name="n")
        input_space = Space([m,n])
    
        """ tuning parameter space """
        #rls_method = Categoricalnorm (["blendenpik", "lsrn", "newtonsketch"], transform="onehot", name="rls_method")
        #sketch_operator = Categoricalnorm (["sjlt", "less_uniform"], transform="onehot", name="sketch_operator")
        sampling_factor = Real(1.0, 10.0, transform="normalize", name="sampling_factor")
        vec_nnz = Integer(1, 100, transform="normalize", name="vec_nnz")
        safety_exponent = Integer(0, 4, transform="normalize", name="safety_exponent")
        #parameter_space = Space([rls_method,sketch_operator,sampling_factor,vec_nnz,safety_exponent])
        parameter_space = Space([sampling_factor,vec_nnz,safety_exponent])
    
        if NS == 1:
            rls_method = "blendenpik"
            sketch_operator = "sjlt"
        elif NS == 2:
            best_func_eval = LoadBestSourceFunctionEvaluation()
            rls_method = best_func_eval["tuning_parameter"]["rls_method"]
            sketch_operator = best_func_eval["tuning_parameter"]["sketch_operator"]
        else:
            def LoadCategory():
                num_categories = 6
                evaluations_per_category = [[] for i in range(num_categories)]
                with open("lhsmdu.db/LHSMDU-SEARCH-failure_handling_"+str(source_failure_handling)+"-n_rows_"+str(source_n_rows)+"-n_cols_"+str(source_n_cols)+"-mattype_"+str(source_mattype)+"-batch_num_1.json") as f_in:
                    function_evaluations = json.load(f_in)["func_eval"][0:100]
                    for i in range(len(function_evaluations)):
                        func_eval = function_evaluations[i]
                        wall_clock_time = np.average(func_eval["additional_output"]["parla_times"])
                        normalized_residual_error_to_Axstar = np.average(func_eval["additional_output"]["normalized_residual_errors_to_Axstar"])
                        if func_eval["tuning_parameter"]["rls_method"] == "blendenpik" and func_eval["tuning_parameter"]["sketch_operator"] == "sjlt":
                            category = 0
                        elif func_eval["tuning_parameter"]["rls_method"] == "lsrn" and func_eval["tuning_parameter"]["sketch_operator"] == "sjlt":
                            category = 1
                        elif func_eval["tuning_parameter"]["rls_method"] == "newtonsketch" and func_eval["tuning_parameter"]["sketch_operator"] == "sjlt":
                            category = 2
                        elif func_eval["tuning_parameter"]["rls_method"] == "blendenpik" and func_eval["tuning_parameter"]["sketch_operator"] == "less_uniform":
                            category = 3
                        elif func_eval["tuning_parameter"]["rls_method"] == "lsrn" and func_eval["tuning_parameter"]["sketch_operator"] == "less_uniform":
                            category = 4
                        elif func_eval["tuning_parameter"]["rls_method"] == "newtonsketch" and func_eval["tuning_parameter"]["sketch_operator"] == "less_uniform":
                            category = 5
                        evaluations_per_category[category].append(func_eval["evaluation_result"]["obj"])
                average_evaluations_per_category = [np.mean(evaluations_per_category[i]) for i in range(len(evaluations_per_category))]
                print ("average_evaluations_per_category: ", average_evaluations_per_category)
                best_category = np.argmin(average_evaluations_per_category)
                print ("best_category: ", best_category)

            category = LoadCategory()
    
            if category == 0:
                rls_method = "blendenpik"
                sketch_operator = "sjlt"
            elif category == 1:
                rls_method = "lsrn"
                sketch_operator = "sjlt"
            elif category == 2:
                rls_method = "newtonsketch"
                sketch_operator = "sjlt"
            elif category == 3:
                rls_method = "blendenpik"
                sketch_operator = "less_uniform"
            elif category == 4:
                rls_method = "lsrn"
                sketch_operator = "less_uniform"
            elif category == 5:
                rls_method = "newtonsketch"
                sketch_operator = "less_uniform"
    
        """ ouput space """
        if failure_handling == "skip":
            obj = Real(float(0.0), float(900.0), name="obj", optimize=True)
        else:
            obj = Real(float("-Inf"), float("Inf"), name="obj", optimize=True)
        output_space = Space([obj])
    
        """ constant variables """
        constants = {
            "rls_method": rls_method,
            "sketch_operator": sketch_operator,
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
    
        options['sample_class'] = 'SampleLHSMDU'
        options['sample_random_seed'] = batch_num
        options['model_class'] = 'Model_GPy_LCM'
        options['model_random_seed'] = batch_num
        options['search_class'] = 'SearchPyGMO'
        options['search_random_seed'] = batch_num
    
        options['TLA_method'] = 'LCM'
    
        if failure_handling == "skip":
            options['model_output_constraint'] = 'Ignore'
        options.validate(computer=computer)
    
        """ run gptune """
        giventask = [[n_rows, n_cols]]
        NI=len(giventask)
        #NS=nrun
    
        gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
    
        source_function_evaluations, best_func_eval = LoadSourceFunctionEvaluations(rls_method, sketch_operator)
        #print ("source_function_evaluations: ", source_function_evaluations)
        if NS == 1:
            P = [[5.0, 50, 0]]
            gt.EvaluateObjective(T=[n_rows, n_cols], P=P)
        elif NS == 2:
            P = [[best_func_eval["tuning_parameter"]["sampling_factor"], best_func_eval["tuning_parameter"]["vec_nnz"], best_func_eval["tuning_parameter"]["safety_exponent"]]]
            gt.EvaluateObjective(T=[n_rows, n_cols], P=P)
        else:
            (data, modeler, stats) = gt.TLA_I(NS=NS, Tnew=giventask, source_function_evaluations=source_function_evaluations)
            if NS == nrun:
                return (data, modeler, stats)

def RunTLA_SourceGreedyEpsilon(nrun, epsilon):

    global A, b

    global seed

    global mattype
    global n_rows
    global n_cols

    global source_mattype
    global source_n_rows
    global source_n_cols

    global nthreads
    global x_star
    global direct_time
    global direct_relative_residual

    global batch_num

    global reference_normalized_residual_error_to_Axstar

    global failure_handling
    global source_failure_handling
    global mab_policy

    print ("ASDFASDF")

    if os.path.exists("gptune_tla.db/GPTUNE-TLA_SourceGreedyEpsilon"+str(epsilon)+"-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num)+".json"):
        print ("experiment was performed")
        print ("gptune_tla.db/GPTUNE-TLA_SourceGreedyEpsilon"+str(epsilon)+"-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num)+".json")
        return

    for NS in range(1, nrun+1, 1):
        """ tuning meta information """
        tuning_metadata = {
            "tuning_problem_name": "GPTUNE-TLA_SourceGreedyEpsilon"+str(epsilon)+"-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num),
            "historydb_path": "gptune_tla.db",
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
    
        niter = 5
    
        """ input space """
        m = Integer(1000, 100000, transform="normalize", name="m")
        n = Integer(1000, 10000, transform="normalize", name="n")
        input_space = Space([m,n])
    
        """ tuning parameter space """
        #rls_method = Categoricalnorm (["blendenpik", "lsrn", "newtonsketch"], transform="onehot", name="rls_method")
        #sketch_operator = Categoricalnorm (["sjlt", "less_uniform"], transform="onehot", name="sketch_operator")
        sampling_factor = Real(1.0, 10.0, transform="normalize", name="sampling_factor")
        vec_nnz = Integer(1, 100, transform="normalize", name="vec_nnz")
        safety_exponent = Integer(0, 4, transform="normalize", name="safety_exponent")
        #parameter_space = Space([rls_method,sketch_operator,sampling_factor,vec_nnz,safety_exponent])
        parameter_space = Space([sampling_factor,vec_nnz,safety_exponent])
    
        if NS == 1:
            rls_method = "blendenpik"
            sketch_operator = "sjlt"
        elif NS == 2:
            best_func_eval = LoadBestSourceFunctionEvaluation()
            rls_method = best_func_eval["tuning_parameter"]["rls_method"]
            sketch_operator = best_func_eval["tuning_parameter"]["sketch_operator"]
        else:
            def LoadCategory(epsilon, batch_num):
                np.random.seed(batch_num)
                if np.random.rand() < epsilon:
                    random_category = np.random.randint(6)
                    return random_category
                else:
                    num_categories = 6
                    evaluations_per_category = [[] for i in range(num_categories)]
                    with open("lhsmdu.db/LHSMDU-SEARCH-failure_handling_"+str(source_failure_handling)+"-n_rows_"+str(source_n_rows)+"-n_cols_"+str(source_n_cols)+"-mattype_"+str(source_mattype)+"-batch_num_1.json") as f_in:
                        function_evaluations = json.load(f_in)["func_eval"][0:100]
                        for i in range(len(function_evaluations)):
                            func_eval = function_evaluations[i]
                            wall_clock_time = np.average(func_eval["additional_output"]["parla_times"])
                            normalized_residual_error_to_Axstar = np.average(func_eval["additional_output"]["normalized_residual_errors_to_Axstar"])
                            if func_eval["tuning_parameter"]["rls_method"] == "blendenpik" and func_eval["tuning_parameter"]["sketch_operator"] == "sjlt":
                                category = 0
                            elif func_eval["tuning_parameter"]["rls_method"] == "lsrn" and func_eval["tuning_parameter"]["sketch_operator"] == "sjlt":
                                category = 1
                            elif func_eval["tuning_parameter"]["rls_method"] == "newtonsketch" and func_eval["tuning_parameter"]["sketch_operator"] == "sjlt":
                                category = 2
                            elif func_eval["tuning_parameter"]["rls_method"] == "blendenpik" and func_eval["tuning_parameter"]["sketch_operator"] == "less_uniform":
                                category = 3
                            elif func_eval["tuning_parameter"]["rls_method"] == "lsrn" and func_eval["tuning_parameter"]["sketch_operator"] == "less_uniform":
                                category = 4
                            elif func_eval["tuning_parameter"]["rls_method"] == "newtonsketch" and func_eval["tuning_parameter"]["sketch_operator"] == "less_uniform":
                                category = 5
                            evaluations_per_category[category].append(func_eval["evaluation_result"]["obj"])
                    average_evaluations_per_category = [np.mean(evaluations_per_category[i]) for i in range(len(evaluations_per_category))]
                    print ("average_evaluations_per_category: ", average_evaluations_per_category)
                    best_category = np.argmin(average_evaluations_per_category)
                    print ("best_category: ", best_category)

            category = LoadCategory(epsilon, batch_num)
    
            if category == 0:
                rls_method = "blendenpik"
                sketch_operator = "sjlt"
            elif category == 1:
                rls_method = "lsrn"
                sketch_operator = "sjlt"
            elif category == 2:
                rls_method = "newtonsketch"
                sketch_operator = "sjlt"
            elif category == 3:
                rls_method = "blendenpik"
                sketch_operator = "less_uniform"
            elif category == 4:
                rls_method = "lsrn"
                sketch_operator = "less_uniform"
            elif category == 5:
                rls_method = "newtonsketch"
                sketch_operator = "less_uniform"
    
        """ ouput space """
        if failure_handling == "skip":
            obj = Real(float(0.0), float(900.0), name="obj", optimize=True)
        else:
            obj = Real(float("-Inf"), float("Inf"), name="obj", optimize=True)
        output_space = Space([obj])
    
        """ constant variables """
        constants = {
            "rls_method": rls_method,
            "sketch_operator": sketch_operator,
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
    
        options['sample_class'] = 'SampleLHSMDU'
        options['sample_random_seed'] = batch_num
        options['model_class'] = 'Model_GPy_LCM'
        options['model_random_seed'] = batch_num
        options['search_class'] = 'SearchPyGMO'
        options['search_random_seed'] = batch_num
    
        options['TLA_method'] = 'LCM'
    
        if failure_handling == "skip":
            options['model_output_constraint'] = 'Ignore'
        options.validate(computer=computer)
    
        """ run gptune """
        giventask = [[n_rows, n_cols]]
        NI=len(giventask)
        #NS=nrun
    
        gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
    
        source_function_evaluations, best_func_eval = LoadSourceFunctionEvaluations(rls_method, sketch_operator)
        #print ("source_function_evaluations: ", source_function_evaluations)
        if NS == 1:
            P = [[5.0, 50, 0]]
            gt.EvaluateObjective(T=[n_rows, n_cols], P=P)
        elif NS == 2:
            P = [[best_func_eval["tuning_parameter"]["sampling_factor"], best_func_eval["tuning_parameter"]["vec_nnz"], best_func_eval["tuning_parameter"]["safety_exponent"]]]
            gt.EvaluateObjective(T=[n_rows, n_cols], P=P)
        else:
            (data, modeler, stats) = gt.TLA_I(NS=NS, Tnew=giventask, source_function_evaluations=source_function_evaluations)
            if NS == nrun:
                return (data, modeler, stats)

def RunTLA_TargetGreedy(nrun, trials):

    global A, b
    global seed

    global mattype
    global n_rows
    global n_cols

    global source_mattype
    global source_n_rows
    global source_n_cols

    global nthreads
    global x_star
    global direct_time
    global direct_relative_residual

    global batch_num
    global reference_normalized_residual_error_to_Axstar

    global failure_handling
    global source_failure_handling
    global mab_policy

    if os.path.exists("gptune_tla.db/GPTUNE-TLA_TargetGreedyTrials"+str(trials)+"-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num)+".json"):
        print ("experiment was already performed")
        print ("gptune_tla.db/GPTUNE-TLA_TargetGreedyTrials"+str(trials)+"-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num)+".json")
        return

    """ tuning meta information """
    tuning_metadata = {
        "tuning_problem_name": "GPTUNE-TLA_TargetGreedyTrials"+str(trials)+"-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num),
        "historydb_path": "gptune_tla.db",
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

    for NS in range(1, 3, 1):
        """ input space """
        m = Integer(1000, 100000, transform="normalize", name="m")
        n = Integer(1000, 10000, transform="normalize", name="n")
        input_space = Space([m,n])
    
        """ tuning parameter space """
        #rls_method = Categoricalnorm (["blendenpik", "lsrn", "newtonsketch"], transform="onehot", name="rls_method")
        #sketch_operator = Categoricalnorm (["sjlt", "less_uniform"], transform="onehot", name="sketch_operator")
        sampling_factor = Real(1.0, 10.0, transform="normalize", name="sampling_factor")
        vec_nnz = Integer(1, 100, transform="normalize", name="vec_nnz")
        safety_exponent = Integer(0, 4, transform="normalize", name="safety_exponent")
        #parameter_space = Space([rls_method,sketch_operator,sampling_factor,vec_nnz,safety_exponent])
        parameter_space = Space([sampling_factor,vec_nnz,safety_exponent])
    
        if NS == 1:
            rls_method = "blendenpik"
            sketch_operator = "sjlt"
        elif NS == 2:
            best_func_eval = LoadBestSourceFunctionEvaluation()
            rls_method = best_func_eval["tuning_parameter"]["rls_method"]
            sketch_operator = best_func_eval["tuning_parameter"]["sketch_operator"]
    
        """ ouput space """
        if failure_handling == "skip":
            obj = Real(float(0.0), float(900.0), name="obj", optimize=True)
        else:
            obj = Real(float("-Inf"), float("Inf"), name="obj", optimize=True)
        output_space = Space([obj])
    
        """ constant variables """
        niter = 5
        constants = {
            "rls_method": rls_method,
            "sketch_operator": sketch_operator,
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
    
        options['sample_class'] = 'SampleLHSMDU'
        options['sample_random_seed'] = batch_num
        options['model_class'] = 'Model_GPy_LCM'
        options['model_random_seed'] = batch_num
        options['search_class'] = 'SearchPyGMO'
        options['search_random_seed'] = batch_num
    
        options['TLA_method'] = 'LCM'
    
        if failure_handling == "skip":
            options['model_output_constraint'] = 'Ignore'
        options.validate(computer=computer)
    
        """ run gptune """
        giventask = [[n_rows, n_cols]]
        NI=len(giventask)
    
        gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
        if NS == 1:
            P = [[5.0, 50, 0]]
            gt.EvaluateObjective(T=[n_rows, n_cols], P=P)
        elif NS == 2:
            P = [[best_func_eval["tuning_parameter"]["sampling_factor"], best_func_eval["tuning_parameter"]["vec_nnz"], best_func_eval["tuning_parameter"]["safety_exponent"]]]
            gt.EvaluateObjective(T=[n_rows, n_cols], P=P)

    num_categories = 6
    np.random.seed(batch_num)
    for NS in range(3, 3+(num_categories*trials), num_categories):
        sampling_factor_ = (9.0)*np.random.rand()+1.0
        vec_nnz_ = np.random.randint(1, 100)
        safety_exponent_ = np.random.randint(0, 4)

        for category in range(num_categories):
            """ input space """
            m = Integer(1000, 100000, transform="normalize", name="m")
            n = Integer(1000, 10000, transform="normalize", name="n")
            input_space = Space([m,n])

            """ tuning parameter space """
            #rls_method = Categoricalnorm (["blendenpik", "lsrn", "newtonsketch"], transform="onehot", name="rls_method")
            #sketch_operator = Categoricalnorm (["sjlt", "less_uniform"], transform="onehot", name="sketch_operator")
            sampling_factor = Real(1.0, 10.0, transform="normalize", name="sampling_factor")
            vec_nnz = Integer(1, 100, transform="normalize", name="vec_nnz")
            safety_exponent = Integer(0, 4, transform="normalize", name="safety_exponent")
            #parameter_space = Space([rls_method,sketch_operator,sampling_factor,vec_nnz,safety_exponent])
            parameter_space = Space([sampling_factor,vec_nnz,safety_exponent])

            if category == 0:
                rls_method = "blendenpik"
                sketch_operator = "sjlt"
            elif category == 1:
                rls_method = "lsrn"
                sketch_operator = "sjlt"
            elif category == 2:
                rls_method = "newtonsketch"
                sketch_operator = "sjlt"
            elif category == 3:
                rls_method = "blendenpik"
                sketch_operator = "less_uniform"
            elif category == 4:
                rls_method = "lsrn"
                sketch_operator = "less_uniform"
            elif category == 5:
                rls_method = "newtonsketch"
                sketch_operator = "less_uniform"
 
            """ ouput space """
            if failure_handling == "skip":
                obj = Real(float(0.0), float(900.0), name="obj", optimize=True)
            else:
                obj = Real(float("-Inf"), float("Inf"), name="obj", optimize=True)
            output_space = Space([obj])
        
            """ constant variables """
            niter = 5
            constants = {
                "rls_method": rls_method,
                "sketch_operator": sketch_operator,
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
        
            options['sample_class'] = 'SampleLHSMDU'
            options['sample_random_seed'] = batch_num
            options['model_class'] = 'Model_GPy_LCM'
            options['model_random_seed'] = batch_num
            options['search_class'] = 'SearchPyGMO'
            options['search_random_seed'] = batch_num
        
            options['TLA_method'] = 'LCM'
        
            if failure_handling == "skip":
                options['model_output_constraint'] = 'Ignore'
            options.validate(computer=computer)
        
            """ run gptune """
            giventask = [[n_rows, n_cols]]
            NI=len(giventask)
        
            gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
            P = [[sampling_factor_, vec_nnz_, safety_exponent_]]
            gt.EvaluateObjective(T=[n_rows, n_cols], P=P)

    num_categories = 6
    evaluations_per_category = [[] for i in range(num_categories)]
    with open("gptune_tla.db/GPTUNE-TLA_TargetGreedyTrials"+str(trials)+"-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num)+".json", "r") as f_in:
        function_evaluations = json.load(f_in)["func_eval"][2:]
        for i in range(len(function_evaluations)):
            func_eval = function_evaluations[i]
            wall_clock_time = np.average(func_eval["additional_output"]["parla_times"])
            normalized_residual_error_to_Axstar = np.average(func_eval["additional_output"]["normalized_residual_errors_to_Axstar"])
            if func_eval["constants"]["rls_method"] == "blendenpik" and func_eval["constants"]["sketch_operator"] == "sjlt":
                category = 0
            elif func_eval["constants"]["rls_method"] == "lsrn" and func_eval["constants"]["sketch_operator"] == "sjlt":
                category = 1
            elif func_eval["constants"]["rls_method"] == "newtonsketch" and func_eval["constants"]["sketch_operator"] == "sjlt":
                category = 2
            elif func_eval["constants"]["rls_method"] == "blendenpik" and func_eval["constants"]["sketch_operator"] == "less_uniform":
                category = 3
            elif func_eval["constants"]["rls_method"] == "lsrn" and func_eval["constants"]["sketch_operator"] == "less_uniform":
                category = 4
            elif func_eval["constants"]["rls_method"] == "newtonsketch" and func_eval["constants"]["sketch_operator"] == "less_uniform":
                category = 5
            evaluations_per_category[category].append(func_eval["evaluation_result"]["obj"])
    average_evaluations_per_category = [np.mean(evaluations_per_category[i]) for i in range(len(evaluations_per_category))]
    print ("average_evaluations_per_category: ", average_evaluations_per_category)
    best_category = np.argmin(average_evaluations_per_category)
    print ("best_category: ", best_category)

    category = best_category
 
    for NS in range(3+(num_categories*trials), nrun+1, 1):
        niter = 5

        """ input space """
        m = Integer(1000, 100000, transform="normalize", name="m")
        n = Integer(1000, 10000, transform="normalize", name="n")
        input_space = Space([m,n])

        """ tuning parameter space """
        #rls_method = Categoricalnorm (["blendenpik", "lsrn", "newtonsketch"], transform="onehot", name="rls_method")
        #sketch_operator = Categoricalnorm (["sjlt", "less_uniform"], transform="onehot", name="sketch_operator")
        sampling_factor = Real(1.0, 10.0, transform="normalize", name="sampling_factor")
        vec_nnz = Integer(1, 100, transform="normalize", name="vec_nnz")
        safety_exponent = Integer(0, 4, transform="normalize", name="safety_exponent")
        #parameter_space = Space([rls_method,sketch_operator,sampling_factor,vec_nnz,safety_exponent])
        parameter_space = Space([sampling_factor,vec_nnz,safety_exponent])

        if category == 0:
            rls_method = "blendenpik"
            sketch_operator = "sjlt"
        elif category == 1:
            rls_method = "lsrn"
            sketch_operator = "sjlt"
        elif category == 2:
            rls_method = "newtonsketch"
            sketch_operator = "sjlt"
        elif category == 3:
            rls_method = "blendenpik"
            sketch_operator = "less_uniform"
        elif category == 4:
            rls_method = "lsrn"
            sketch_operator = "less_uniform"
        elif category == 5:
            rls_method = "newtonsketch"
            sketch_operator = "less_uniform"

        """ ouput space """
        if failure_handling == "skip":
            obj = Real(float(0.0), float(900.0), name="obj", optimize=True)
        else:
            obj = Real(float("-Inf"), float("Inf"), name="obj", optimize=True)
        output_space = Space([obj])
    
        """ constant variables """
        constants = {
            "rls_method": rls_method,
            "sketch_operator": sketch_operator,
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
    
        options['sample_class'] = 'SampleLHSMDU'
        options['sample_random_seed'] = batch_num
        options['model_class'] = 'Model_GPy_LCM'
        options['model_random_seed'] = batch_num
        options['search_class'] = 'SearchPyGMO'
        options['search_random_seed'] = batch_num
    
        options['TLA_method'] = 'LCM'
    
        if failure_handling == "skip":
            options['model_output_constraint'] = 'Ignore'
        options.validate(computer=computer)
    
        """ run gptune """
        giventask = [[n_rows, n_cols]]
        NI=len(giventask)
        #NS=nrun
    
        gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
    
        source_function_evaluations, best_func_eval = LoadSourceFunctionEvaluations(rls_method, sketch_operator)
        #print ("source_function_evaluations: ", source_function_evaluations)

        (data, modeler, stats) = gt.TLA_I(NS=NS, Tnew=giventask, source_function_evaluations=source_function_evaluations)
        if NS == nrun:
            return (data, modeler, stats)

def RunTLA_TargetGreedyEpsilon(nrun, trials, epsilon):

    global A, b
    global seed

    global mattype
    global n_rows
    global n_cols

    global source_mattype
    global source_n_rows
    global source_n_cols

    global nthreads
    global x_star
    global direct_time
    global direct_relative_residual

    global batch_num
    global reference_normalized_residual_error_to_Axstar

    global failure_handling
    global source_failure_handling
    global mab_policy

    if os.path.exists("gptune_tla.db/GPTUNE-TLA_TargetGreedyTrials"+str(trials)+"Epsilon"+str(epsilon)+"-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num)+".json"):
        print ("experiment was already performed")
        print ("gptune_tla.db/GPTUNE-TLA_TargetGreedyTrials"+str(trials)+"Epsilon"+str(epsilon)+"-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num)+".json")
        return

    """ tuning meta information """
    tuning_metadata = {
        "tuning_problem_name": "GPTUNE-TLA_TargetGreedyTrials"+str(trials)+"Epsilon"+str(epsilon)+"-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num),
        "historydb_path": "gptune_tla.db",
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

    for NS in range(1, 3, 1):
        """ input space """
        m = Integer(1000, 100000, transform="normalize", name="m")
        n = Integer(1000, 10000, transform="normalize", name="n")
        input_space = Space([m,n])
    
        """ tuning parameter space """
        #rls_method = Categoricalnorm (["blendenpik", "lsrn", "newtonsketch"], transform="onehot", name="rls_method")
        #sketch_operator = Categoricalnorm (["sjlt", "less_uniform"], transform="onehot", name="sketch_operator")
        sampling_factor = Real(1.0, 10.0, transform="normalize", name="sampling_factor")
        vec_nnz = Integer(1, 100, transform="normalize", name="vec_nnz")
        safety_exponent = Integer(0, 4, transform="normalize", name="safety_exponent")
        #parameter_space = Space([rls_method,sketch_operator,sampling_factor,vec_nnz,safety_exponent])
        parameter_space = Space([sampling_factor,vec_nnz,safety_exponent])
    
        if NS == 1:
            rls_method = "blendenpik"
            sketch_operator = "sjlt"
        elif NS == 2:
            best_func_eval = LoadBestSourceFunctionEvaluation()
            rls_method = best_func_eval["tuning_parameter"]["rls_method"]
            sketch_operator = best_func_eval["tuning_parameter"]["sketch_operator"]
    
        """ ouput space """
        if failure_handling == "skip":
            obj = Real(float(0.0), float(900.0), name="obj", optimize=True)
        else:
            obj = Real(float("-Inf"), float("Inf"), name="obj", optimize=True)
        output_space = Space([obj])
    
        """ constant variables """
        niter = 5
        constants = {
            "rls_method": rls_method,
            "sketch_operator": sketch_operator,
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
    
        options['sample_class'] = 'SampleLHSMDU'
        options['sample_random_seed'] = batch_num
        options['model_class'] = 'Model_GPy_LCM'
        options['model_random_seed'] = batch_num
        options['search_class'] = 'SearchPyGMO'
        options['search_random_seed'] = batch_num
    
        options['TLA_method'] = 'LCM'
    
        if failure_handling == "skip":
            options['model_output_constraint'] = 'Ignore'
        options.validate(computer=computer)
    
        """ run gptune """
        giventask = [[n_rows, n_cols]]
        NI=len(giventask)
    
        gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
        if NS == 1:
            P = [[5.0, 50, 0]]
            gt.EvaluateObjective(T=[n_rows, n_cols], P=P)
        elif NS == 2:
            P = [[best_func_eval["tuning_parameter"]["sampling_factor"], best_func_eval["tuning_parameter"]["vec_nnz"], best_func_eval["tuning_parameter"]["safety_exponent"]]]
            gt.EvaluateObjective(T=[n_rows, n_cols], P=P)

    num_categories = 6
    np.random.seed(batch_num)
    for NS in range(3, 3+(num_categories*trials), num_categories):
        sampling_factor_ = (9.0)*np.random.rand()+1.0
        vec_nnz_ = np.random.randint(1, 100)
        safety_exponent_ = np.random.randint(0, 4)

        for category in range(num_categories):
            """ input space """
            m = Integer(1000, 100000, transform="normalize", name="m")
            n = Integer(1000, 10000, transform="normalize", name="n")
            input_space = Space([m,n])

            """ tuning parameter space """
            #rls_method = Categoricalnorm (["blendenpik", "lsrn", "newtonsketch"], transform="onehot", name="rls_method")
            #sketch_operator = Categoricalnorm (["sjlt", "less_uniform"], transform="onehot", name="sketch_operator")
            sampling_factor = Real(1.0, 10.0, transform="normalize", name="sampling_factor")
            vec_nnz = Integer(1, 100, transform="normalize", name="vec_nnz")
            safety_exponent = Integer(0, 4, transform="normalize", name="safety_exponent")
            #parameter_space = Space([rls_method,sketch_operator,sampling_factor,vec_nnz,safety_exponent])
            parameter_space = Space([sampling_factor,vec_nnz,safety_exponent])

            if category == 0:
                rls_method = "blendenpik"
                sketch_operator = "sjlt"
            elif category == 1:
                rls_method = "lsrn"
                sketch_operator = "sjlt"
            elif category == 2:
                rls_method = "newtonsketch"
                sketch_operator = "sjlt"
            elif category == 3:
                rls_method = "blendenpik"
                sketch_operator = "less_uniform"
            elif category == 4:
                rls_method = "lsrn"
                sketch_operator = "less_uniform"
            elif category == 5:
                rls_method = "newtonsketch"
                sketch_operator = "less_uniform"
 
            """ ouput space """
            if failure_handling == "skip":
                obj = Real(float(0.0), float(900.0), name="obj", optimize=True)
            else:
                obj = Real(float("-Inf"), float("Inf"), name="obj", optimize=True)
            output_space = Space([obj])
        
            """ constant variables """
            niter = 5
            constants = {
                "rls_method": rls_method,
                "sketch_operator": sketch_operator,
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
        
            options['sample_class'] = 'SampleLHSMDU'
            options['sample_random_seed'] = batch_num
            options['model_class'] = 'Model_GPy_LCM'
            options['model_random_seed'] = batch_num
            options['search_class'] = 'SearchPyGMO'
            options['search_random_seed'] = batch_num
        
            options['TLA_method'] = 'LCM'
        
            if failure_handling == "skip":
                options['model_output_constraint'] = 'Ignore'
            options.validate(computer=computer)
        
            """ run gptune """
            giventask = [[n_rows, n_cols]]
            NI=len(giventask)
        
            gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
            P = [[sampling_factor_, vec_nnz_, safety_exponent_]]
            gt.EvaluateObjective(T=[n_rows, n_cols], P=P)

    for NS in range(3+(num_categories*trials), nrun+1, 1):
        niter = 5

        """ input space """
        m = Integer(1000, 100000, transform="normalize", name="m")
        n = Integer(1000, 10000, transform="normalize", name="n")
        input_space = Space([m,n])

        """ tuning parameter space """
        #rls_method = Categoricalnorm (["blendenpik", "lsrn", "newtonsketch"], transform="onehot", name="rls_method")
        #sketch_operator = Categoricalnorm (["sjlt", "less_uniform"], transform="onehot", name="sketch_operator")
        sampling_factor = Real(1.0, 10.0, transform="normalize", name="sampling_factor")
        vec_nnz = Integer(1, 100, transform="normalize", name="vec_nnz")
        safety_exponent = Integer(0, 4, transform="normalize", name="safety_exponent")
        #parameter_space = Space([rls_method,sketch_operator,sampling_factor,vec_nnz,safety_exponent])
        parameter_space = Space([sampling_factor,vec_nnz,safety_exponent])

        if np.random.rand() < epsilon:
            category = np.random.randint(6)
            print ("randomly chosen category: ", category)
        else:
            num_categories = 6
            evaluations_per_category = [[] for i in range(num_categories)]
            with open("gptune_tla.db/GPTUNE-TLA_TargetGreedyTrials"+str(trials)+"Epsilon"+str(epsilon)+"-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num)+".json", "r") as f_in:
                function_evaluations = json.load(f_in)["func_eval"][2:]
                for i in range(len(function_evaluations)):
                    func_eval = function_evaluations[i]
                    wall_clock_time = np.average(func_eval["additional_output"]["parla_times"])
                    normalized_residual_error_to_Axstar = np.average(func_eval["additional_output"]["normalized_residual_errors_to_Axstar"])
                    if func_eval["constants"]["rls_method"] == "blendenpik" and func_eval["constants"]["sketch_operator"] == "sjlt":
                        category = 0
                    elif func_eval["constants"]["rls_method"] == "lsrn" and func_eval["constants"]["sketch_operator"] == "sjlt":
                        category = 1
                    elif func_eval["constants"]["rls_method"] == "newtonsketch" and func_eval["constants"]["sketch_operator"] == "sjlt":
                        category = 2
                    elif func_eval["constants"]["rls_method"] == "blendenpik" and func_eval["constants"]["sketch_operator"] == "less_uniform":
                        category = 3
                    elif func_eval["constants"]["rls_method"] == "lsrn" and func_eval["constants"]["sketch_operator"] == "less_uniform":
                        category = 4
                    elif func_eval["constants"]["rls_method"] == "newtonsketch" and func_eval["constants"]["sketch_operator"] == "less_uniform":
                        category = 5
                    evaluations_per_category[category].append(func_eval["evaluation_result"]["obj"])
            average_evaluations_per_category = [np.mean(evaluations_per_category[i]) for i in range(len(evaluations_per_category))]
            print ("average_evaluations_per_category: ", average_evaluations_per_category)
            best_category = np.argmin(average_evaluations_per_category)
            print ("best_category: ", best_category)
            category = best_category
 
        if category == 0:
            rls_method = "blendenpik"
            sketch_operator = "sjlt"
        elif category == 1:
            rls_method = "lsrn"
            sketch_operator = "sjlt"
        elif category == 2:
            rls_method = "newtonsketch"
            sketch_operator = "sjlt"
        elif category == 3:
            rls_method = "blendenpik"
            sketch_operator = "less_uniform"
        elif category == 4:
            rls_method = "lsrn"
            sketch_operator = "less_uniform"
        elif category == 5:
            rls_method = "newtonsketch"
            sketch_operator = "less_uniform"

        """ ouput space """
        if failure_handling == "skip":
            obj = Real(float(0.0), float(900.0), name="obj", optimize=True)
        else:
            obj = Real(float("-Inf"), float("Inf"), name="obj", optimize=True)
        output_space = Space([obj])

        """ constant variables """
        constants = {
            "rls_method": rls_method,
            "sketch_operator": sketch_operator,
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

        options['sample_class'] = 'SampleLHSMDU'
        options['sample_random_seed'] = batch_num
        options['model_class'] = 'Model_GPy_LCM'
        options['model_random_seed'] = batch_num
        options['search_class'] = 'SearchPyGMO'
        options['search_random_seed'] = batch_num

        options['TLA_method'] = 'LCM'

        if failure_handling == "skip":
            options['model_output_constraint'] = 'Ignore'
        options.validate(computer=computer)

        """ run gptune """
        giventask = [[n_rows, n_cols]]
        NI=len(giventask)
        #NS=nrun

        gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))

        source_function_evaluations, best_func_eval = LoadSourceFunctionEvaluations(rls_method, sketch_operator)
        #print ("source_function_evaluations: ", source_function_evaluations)

        (data, modeler, stats) = gt.TLA_I(NS=NS, Tnew=giventask, source_function_evaluations=source_function_evaluations)
        if NS == nrun:
            return (data, modeler, stats)

def RunTLA_UCB(nrun, trials, c_ucb):

    global A, b
    global seed

    global mattype
    global n_rows
    global n_cols

    global source_mattype
    global source_n_rows
    global source_n_cols

    global nthreads
    global x_star
    global direct_time
    global direct_relative_residual

    global batch_num
    global reference_normalized_residual_error_to_Axstar

    global failure_handling
    global source_failure_handling
    global mab_policy

    if os.path.exists("gptune_tla.db/GPTUNE-TLA_UCB_Trials"+str(trials)+"_CUCB_"+str(c_ucb)+"-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num)+".json"):
        print ("experiment was already performed")
        print ("gptune_tla.db/GPTUNE-TLA_UCB_Trials"+str(trials)+"_CUCB_"+str(c_ucb)+"-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num)+".json")
        return

    """ tuning meta information """
    tuning_metadata = {
        "tuning_problem_name": "GPTUNE-TLA_UCB_Trials"+str(trials)+"_CUCB_"+str(c_ucb)+"-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num),
        "historydb_path": "gptune_tla.db",
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

    for NS in range(1, 3, 1):
        """ input space """
        m = Integer(1000, 100000, transform="normalize", name="m")
        n = Integer(1000, 10000, transform="normalize", name="n")
        input_space = Space([m,n])
    
        """ tuning parameter space """
        #rls_method = Categoricalnorm (["blendenpik", "lsrn", "newtonsketch"], transform="onehot", name="rls_method")
        #sketch_operator = Categoricalnorm (["sjlt", "less_uniform"], transform="onehot", name="sketch_operator")
        sampling_factor = Real(1.0, 10.0, transform="normalize", name="sampling_factor")
        vec_nnz = Integer(1, 100, transform="normalize", name="vec_nnz")
        safety_exponent = Integer(0, 4, transform="normalize", name="safety_exponent")
        #parameter_space = Space([rls_method,sketch_operator,sampling_factor,vec_nnz,safety_exponent])
        parameter_space = Space([sampling_factor,vec_nnz,safety_exponent])
    
        if NS == 1:
            rls_method = "blendenpik"
            sketch_operator = "sjlt"
        elif NS == 2:
            best_func_eval = LoadBestSourceFunctionEvaluation()
            rls_method = best_func_eval["tuning_parameter"]["rls_method"]
            sketch_operator = best_func_eval["tuning_parameter"]["sketch_operator"]
    
        """ ouput space """
        if failure_handling == "skip":
            obj = Real(float(0.0), float(900.0), name="obj", optimize=True)
        else:
            obj = Real(float("-Inf"), float("Inf"), name="obj", optimize=True)
        output_space = Space([obj])
    
        """ constant variables """
        niter = 5
        constants = {
            "rls_method": rls_method,
            "sketch_operator": sketch_operator,
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
    
        options['sample_class'] = 'SampleLHSMDU'
        options['sample_random_seed'] = batch_num
        options['model_class'] = 'Model_GPy_LCM'
        options['model_random_seed'] = batch_num
        options['search_class'] = 'SearchPyGMO'
        options['search_random_seed'] = batch_num
    
        options['TLA_method'] = 'LCM'
    
        if failure_handling == "skip":
            options['model_output_constraint'] = 'Ignore'
        options.validate(computer=computer)
    
        """ run gptune """
        giventask = [[n_rows, n_cols]]
        NI=len(giventask)
    
        gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
        if NS == 1:
            P = [[5.0, 50, 0]]
            gt.EvaluateObjective(T=[n_rows, n_cols], P=P)
        elif NS == 2:
            P = [[best_func_eval["tuning_parameter"]["sampling_factor"], best_func_eval["tuning_parameter"]["vec_nnz"], best_func_eval["tuning_parameter"]["safety_exponent"]]]
            gt.EvaluateObjective(T=[n_rows, n_cols], P=P)

    num_categories = 6
    np.random.seed(batch_num)
    for NS in range(3, 3+(num_categories*trials), num_categories):
        sampling_factor_ = (9.0)*np.random.rand()+1.0
        vec_nnz_ = np.random.randint(1, 100)
        safety_exponent_ = np.random.randint(0, 4)

        for category in range(num_categories):
            """ input space """
            m = Integer(1000, 100000, transform="normalize", name="m")
            n = Integer(1000, 10000, transform="normalize", name="n")
            input_space = Space([m,n])

            """ tuning parameter space """
            #rls_method = Categoricalnorm (["blendenpik", "lsrn", "newtonsketch"], transform="onehot", name="rls_method")
            #sketch_operator = Categoricalnorm (["sjlt", "less_uniform"], transform="onehot", name="sketch_operator")
            sampling_factor = Real(1.0, 10.0, transform="normalize", name="sampling_factor")
            vec_nnz = Integer(1, 100, transform="normalize", name="vec_nnz")
            safety_exponent = Integer(0, 4, transform="normalize", name="safety_exponent")
            #parameter_space = Space([rls_method,sketch_operator,sampling_factor,vec_nnz,safety_exponent])
            parameter_space = Space([sampling_factor,vec_nnz,safety_exponent])

            if category == 0:
                rls_method = "blendenpik"
                sketch_operator = "sjlt"
            elif category == 1:
                rls_method = "lsrn"
                sketch_operator = "sjlt"
            elif category == 2:
                rls_method = "newtonsketch"
                sketch_operator = "sjlt"
            elif category == 3:
                rls_method = "blendenpik"
                sketch_operator = "less_uniform"
            elif category == 4:
                rls_method = "lsrn"
                sketch_operator = "less_uniform"
            elif category == 5:
                rls_method = "newtonsketch"
                sketch_operator = "less_uniform"
 
            """ ouput space """
            if failure_handling == "skip":
                obj = Real(float(0.0), float(900.0), name="obj", optimize=True)
            else:
                obj = Real(float("-Inf"), float("Inf"), name="obj", optimize=True)
            output_space = Space([obj])
        
            """ constant variables """
            niter = 5
            constants = {
                "rls_method": rls_method,
                "sketch_operator": sketch_operator,
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
        
            options['sample_class'] = 'SampleLHSMDU'
            options['sample_random_seed'] = batch_num
            options['model_class'] = 'Model_GPy_LCM'
            options['model_random_seed'] = batch_num
            options['search_class'] = 'SearchPyGMO'
            options['search_random_seed'] = batch_num
        
            options['TLA_method'] = 'LCM'
        
            if failure_handling == "skip":
                options['model_output_constraint'] = 'Ignore'
            options.validate(computer=computer)
        
            """ run gptune """
            giventask = [[n_rows, n_cols]]
            NI=len(giventask)
        
            gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
            P = [[sampling_factor_, vec_nnz_, safety_exponent_]]
            gt.EvaluateObjective(T=[n_rows, n_cols], P=P)

    for NS in range(3+(num_categories*trials), nrun+1, 1):
        niter = 5

        """ input space """
        m = Integer(1000, 100000, transform="normalize", name="m")
        n = Integer(1000, 10000, transform="normalize", name="n")
        input_space = Space([m,n])

        """ tuning parameter space """
        #rls_method = Categoricalnorm (["blendenpik", "lsrn", "newtonsketch"], transform="onehot", name="rls_method")
        #sketch_operator = Categoricalnorm (["sjlt", "less_uniform"], transform="onehot", name="sketch_operator")
        sampling_factor = Real(1.0, 10.0, transform="normalize", name="sampling_factor")
        vec_nnz = Integer(1, 100, transform="normalize", name="vec_nnz")
        safety_exponent = Integer(0, 4, transform="normalize", name="safety_exponent")
        #parameter_space = Space([rls_method,sketch_operator,sampling_factor,vec_nnz,safety_exponent])
        parameter_space = Space([sampling_factor,vec_nnz,safety_exponent])

        num_categories = 6
        evaluations_per_category = [[] for i in range(num_categories)]
        with open("gptune_tla.db/GPTUNE-TLA_UCB_Trials"+str(trials)+"_CUCB_"+str(c_ucb)+"-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num)+".json") as f_in:
            function_evaluations = json.load(f_in)["func_eval"][2:]
            for i in range(len(function_evaluations)):
                func_eval = function_evaluations[i]
                wall_clock_time = np.average(func_eval["additional_output"]["parla_times"])
                normalized_residual_error_to_Axstar = np.average(func_eval["additional_output"]["normalized_residual_errors_to_Axstar"])
                if func_eval["constants"]["rls_method"] == "blendenpik" and func_eval["constants"]["sketch_operator"] == "sjlt":
                    category = 0
                elif func_eval["constants"]["rls_method"] == "lsrn" and func_eval["constants"]["sketch_operator"] == "sjlt":
                    category = 1
                elif func_eval["constants"]["rls_method"] == "newtonsketch" and func_eval["constants"]["sketch_operator"] == "sjlt":
                    category = 2
                elif func_eval["constants"]["rls_method"] == "blendenpik" and func_eval["constants"]["sketch_operator"] == "less_uniform":
                    category = 3
                elif func_eval["constants"]["rls_method"] == "lsrn" and func_eval["constants"]["sketch_operator"] == "less_uniform":
                    category = 4
                elif func_eval["constants"]["rls_method"] == "newtonsketch" and func_eval["constants"]["sketch_operator"] == "less_uniform":
                    category = 5
                evaluations_per_category[category].append(func_eval["evaluation_result"]["obj"])
        num_evals_per_category = [len(evaluations_per_category[i]) for i in range(len(evaluations_per_category))]
        average_evaluations_per_category = []
        for i in range(len(evaluations_per_category)):
            if len(evaluations_per_category[i]) == 0:
                average_evaluations_per_category.append(0)
            else:
                average_evaluations_per_category.append(np.mean(evaluations_per_category[i]))
        print ("average_evaluations_per_category: ", average_evaluations_per_category)
        #average_evaluations_per_category = [np.mean(evaluations_per_category[i]) for i in range(len(evaluations_per_category))]
        if np.sum(num_evals_per_category) == 0:
            rewards = np.zeros(num_categories)
        else:
            rewards = np.abs((np.max(average_evaluations_per_category)-average_evaluations_per_category)/(np.max(average_evaluations_per_category)-np.min(average_evaluations_per_category)))
        print ("considered_previous_evaluations_in_total: ", NS-3)
        balance = c_ucb * np.sqrt(np.log(NS-3)/np.array(num_evals_per_category))
        print ("rewards: ", rewards)
        print ("balance: ", balance)
        ucb_values = rewards + balance
        best_category = np.argmax(ucb_values)
        print ("best_category: ", best_category)
        category = best_category
 
        if category == 0:
            rls_method = "blendenpik"
            sketch_operator = "sjlt"
        elif category == 1:
            rls_method = "lsrn"
            sketch_operator = "sjlt"
        elif category == 2:
            rls_method = "newtonsketch"
            sketch_operator = "sjlt"
        elif category == 3:
            rls_method = "blendenpik"
            sketch_operator = "less_uniform"
        elif category == 4:
            rls_method = "lsrn"
            sketch_operator = "less_uniform"
        elif category == 5:
            rls_method = "newtonsketch"
            sketch_operator = "less_uniform"

        """ ouput space """
        if failure_handling == "skip":
            obj = Real(float(0.0), float(900.0), name="obj", optimize=True)
        else:
            obj = Real(float("-Inf"), float("Inf"), name="obj", optimize=True)
        output_space = Space([obj])

        """ constant variables """
        constants = {
            "rls_method": rls_method,
            "sketch_operator": sketch_operator,
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

        options['sample_class'] = 'SampleLHSMDU'
        options['sample_random_seed'] = batch_num
        options['model_class'] = 'Model_GPy_LCM'
        options['model_random_seed'] = batch_num
        options['search_class'] = 'SearchPyGMO'
        options['search_random_seed'] = batch_num

        options['TLA_method'] = 'LCM'

        if failure_handling == "skip":
            options['model_output_constraint'] = 'Ignore'
        options.validate(computer=computer)

        """ run gptune """
        giventask = [[n_rows, n_cols]]
        NI=len(giventask)
        #NS=nrun

        gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))

        source_function_evaluations, best_func_eval = LoadSourceFunctionEvaluations(rls_method, sketch_operator)
        #print ("source_function_evaluations: ", source_function_evaluations)

        (data, modeler, stats) = gt.TLA_I(NS=NS, Tnew=giventask, source_function_evaluations=source_function_evaluations)
        if NS == nrun:
            return (data, modeler, stats)

def RunTLA_HUCB(nrun, trials, c_ucb, c_hist):

    global A, b
    global seed

    global mattype
    global n_rows
    global n_cols

    global source_mattype
    global source_n_rows
    global source_n_cols

    global nthreads
    global x_star
    global direct_time
    global direct_relative_residual

    global batch_num
    global reference_normalized_residual_error_to_Axstar

    global failure_handling
    global source_failure_handling
    global mab_policy

    if os.path.exists("gptune_tla.db/GPTUNE-TLA_HUCB_Trials"+str(trials)+"_CUCB_"+str(c_ucb)+"_CHIST_"+str(c_hist)+"-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num)+".json"):
        #print ("experiment was already performed")
        print ("gptune_tla.db/GPTUNE-TLA_HUCB_Trials"+str(trials)+"_CUCB_"+str(c_ucb)+"_CHIST_"+str(c_hist)+"-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num)+".json")
        with open("gptune_tla.db/GPTUNE-TLA_HUCB_Trials"+str(trials)+"_CUCB_"+str(c_ucb)+"_CHIST_"+str(c_hist)+"-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num)+".json") as f_in:
            historical_function_evaluations = json.load(f_in)["func_eval"]
            NS_hist = len(historical_function_evaluations)
            NS = NS_hist + 1
            reference_normalized_residual_error_to_Axstar = np.average(historical_function_evaluations[0]["additional_output"]["normalized_residual_errors_to_Axstar"])
            print ("reference_normalized_residual_error_to_Axstar: ", reference_normalized_residual_error_to_Axstar)
            #reference_normalized_residual_error_to_Axstar = json.load(f_in)
    else:
        NS_hist = 0
        NS = NS_hist+1

    print ("Number of oreviously collected function evaluations: ", NS_hist)
    print ("NS: ", NS)

    """ tuning meta information """
    tuning_metadata = {
        "tuning_problem_name": "GPTUNE-TLA_HUCB_Trials"+str(trials)+"_CUCB_"+str(c_ucb)+"_CHIST_"+str(c_hist)+"-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num),
        "historydb_path": "gptune_tla.db",
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

    #for NS in range(1, 3, 1):
    while NS < 3:
        print ("NS: ", NS)
        print ("Start tuning for NS: ", NS)

        """ input space """
        m = Integer(1000, 100000, transform="normalize", name="m")
        n = Integer(1000, 10000, transform="normalize", name="n")
        input_space = Space([m,n])
    
        """ tuning parameter space """
        #rls_method = Categoricalnorm (["blendenpik", "lsrn", "newtonsketch"], transform="onehot", name="rls_method")
        #sketch_operator = Categoricalnorm (["sjlt", "less_uniform"], transform="onehot", name="sketch_operator")
        sampling_factor = Real(1.0, 10.0, transform="normalize", name="sampling_factor")
        vec_nnz = Integer(1, 100, transform="normalize", name="vec_nnz")
        safety_exponent = Integer(0, 4, transform="normalize", name="safety_exponent")
        #parameter_space = Space([rls_method,sketch_operator,sampling_factor,vec_nnz,safety_exponent])
        parameter_space = Space([sampling_factor,vec_nnz,safety_exponent])
    
        if NS == 1:
            rls_method = "blendenpik"
            sketch_operator = "sjlt"
        elif NS == 2:
            best_func_eval = LoadBestSourceFunctionEvaluation()
            rls_method = best_func_eval["tuning_parameter"]["rls_method"]
            sketch_operator = best_func_eval["tuning_parameter"]["sketch_operator"]
    
        """ ouput space """
        if failure_handling == "skip":
            obj = Real(float(0.0), float(900.0), name="obj", optimize=True)
        else:
            obj = Real(float("-Inf"), float("Inf"), name="obj", optimize=True)
        output_space = Space([obj])
    
        """ constant variables """
        niter = 5
        constants = {
            "rls_method": rls_method,
            "sketch_operator": sketch_operator,
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
    
        options['sample_class'] = 'SampleLHSMDU'
        options['sample_random_seed'] = batch_num
        options['model_class'] = 'Model_GPy_LCM'
        options['model_random_seed'] = batch_num
        options['search_class'] = 'SearchPyGMO'
        options['search_random_seed'] = batch_num
    
        options['TLA_method'] = 'LCM'
    
        if failure_handling == "skip":
            options['model_output_constraint'] = 'Ignore'
        options.validate(computer=computer)
    
        """ run gptune """
        giventask = [[n_rows, n_cols]]
        NI=len(giventask)
    
        gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
        if NS == 1:
            P = [[5.0, 50, 0]]
            gt.EvaluateObjective(T=[n_rows, n_cols], P=P)
        elif NS == 2:
            P = [[best_func_eval["tuning_parameter"]["sampling_factor"], best_func_eval["tuning_parameter"]["vec_nnz"], best_func_eval["tuning_parameter"]["safety_exponent"]]]
            gt.EvaluateObjective(T=[n_rows, n_cols], P=P)

        NS += 1

    num_categories = 6
    np.random.seed(batch_num)

    #for NS in range(3, 3+(num_categories*trials), num_categories):
    while NS < 3+(num_categories*trials):
        print ("NS: ", NS)
        print ("Start tuning for NS: ", NS)

        sampling_factor_ = (9.0)*np.random.rand()+1.0
        vec_nnz_ = np.random.randint(1, 100)
        safety_exponent_ = np.random.randint(0, 4)

        for category in range(num_categories):
            """ input space """
            m = Integer(1000, 100000, transform="normalize", name="m")
            n = Integer(1000, 10000, transform="normalize", name="n")
            input_space = Space([m,n])

            """ tuning parameter space """
            #rls_method = Categoricalnorm (["blendenpik", "lsrn", "newtonsketch"], transform="onehot", name="rls_method")
            #sketch_operator = Categoricalnorm (["sjlt", "less_uniform"], transform="onehot", name="sketch_operator")
            sampling_factor = Real(1.0, 10.0, transform="normalize", name="sampling_factor")
            vec_nnz = Integer(1, 100, transform="normalize", name="vec_nnz")
            safety_exponent = Integer(0, 4, transform="normalize", name="safety_exponent")
            #parameter_space = Space([rls_method,sketch_operator,sampling_factor,vec_nnz,safety_exponent])
            parameter_space = Space([sampling_factor,vec_nnz,safety_exponent])

            if category == 0:
                rls_method = "blendenpik"
                sketch_operator = "sjlt"
            elif category == 1:
                rls_method = "lsrn"
                sketch_operator = "sjlt"
            elif category == 2:
                rls_method = "newtonsketch"
                sketch_operator = "sjlt"
            elif category == 3:
                rls_method = "blendenpik"
                sketch_operator = "less_uniform"
            elif category == 4:
                rls_method = "lsrn"
                sketch_operator = "less_uniform"
            elif category == 5:
                rls_method = "newtonsketch"
                sketch_operator = "less_uniform"
 
            """ ouput space """
            if failure_handling == "skip":
                obj = Real(float(0.0), float(900.0), name="obj", optimize=True)
            else:
                obj = Real(float("-Inf"), float("Inf"), name="obj", optimize=True)
            output_space = Space([obj])
        
            """ constant variables """
            niter = 5
            constants = {
                "rls_method": rls_method,
                "sketch_operator": sketch_operator,
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
        
            options['sample_class'] = 'SampleLHSMDU'
            options['sample_random_seed'] = batch_num
            options['model_class'] = 'Model_GPy_LCM'
            options['model_random_seed'] = batch_num
            options['search_class'] = 'SearchPyGMO'
            options['search_random_seed'] = batch_num
        
            options['TLA_method'] = 'LCM'
        
            if failure_handling == "skip":
                options['model_output_constraint'] = 'Ignore'
            options.validate(computer=computer)
        
            """ run gptune """
            giventask = [[n_rows, n_cols]]
            NI=len(giventask)
        
            gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))
            P = [[sampling_factor_, vec_nnz_, safety_exponent_]]
            gt.EvaluateObjective(T=[n_rows, n_cols], P=P)

        NS += num_categories

    #for NS in range(3+(num_categories*trials), nrun+1, 1):
    while NS < nrun+1:
        print ("NS: ", NS)
        print ("Start tuning for NS: ", NS)

        niter = 5

        """ input space """
        m = Integer(1000, 100000, transform="normalize", name="m")
        n = Integer(1000, 10000, transform="normalize", name="n")
        input_space = Space([m,n])

        """ tuning parameter space """
        #rls_method = Categoricalnorm (["blendenpik", "lsrn", "newtonsketch"], transform="onehot", name="rls_method")
        #sketch_operator = Categoricalnorm (["sjlt", "less_uniform"], transform="onehot", name="sketch_operator")
        sampling_factor = Real(1.0, 10.0, transform="normalize", name="sampling_factor")
        vec_nnz = Integer(1, 100, transform="normalize", name="vec_nnz")
        safety_exponent = Integer(0, 4, transform="normalize", name="safety_exponent")
        #parameter_space = Space([rls_method,sketch_operator,sampling_factor,vec_nnz,safety_exponent])
        parameter_space = Space([sampling_factor,vec_nnz,safety_exponent])

        num_categories = 6

        source_evaluations_per_category = [[] for i in range(num_categories)]
        with open("lhsmdu.db/LHSMDU-SEARCH-failure_handling_"+str(source_failure_handling)+"-n_rows_"+str(source_n_rows)+"-n_cols_"+str(source_n_cols)+"-mattype_"+str(source_mattype)+"-batch_num_1.json") as f_in:
            function_evaluations = json.load(f_in)["func_eval"][0:100]
            for i in range(len(function_evaluations)):
                func_eval = function_evaluations[i]
                wall_clock_time = np.average(func_eval["additional_output"]["parla_times"])
                normalized_residual_error_to_Axstar = np.average(func_eval["additional_output"]["normalized_residual_errors_to_Axstar"])
                if func_eval["tuning_parameter"]["rls_method"] == "blendenpik" and func_eval["tuning_parameter"]["sketch_operator"] == "sjlt":
                    category = 0
                elif func_eval["tuning_parameter"]["rls_method"] == "lsrn" and func_eval["tuning_parameter"]["sketch_operator"] == "sjlt":
                    category = 1
                elif func_eval["tuning_parameter"]["rls_method"] == "newtonsketch" and func_eval["tuning_parameter"]["sketch_operator"] == "sjlt":
                    category = 2
                elif func_eval["tuning_parameter"]["rls_method"] == "blendenpik" and func_eval["tuning_parameter"]["sketch_operator"] == "less_uniform":
                    category = 3
                elif func_eval["tuning_parameter"]["rls_method"] == "lsrn" and func_eval["tuning_parameter"]["sketch_operator"] == "less_uniform":
                    category = 4
                elif func_eval["tuning_parameter"]["rls_method"] == "newtonsketch" and func_eval["tuning_parameter"]["sketch_operator"] == "less_uniform":
                    category = 5
                source_evaluations_per_category[category].append(func_eval["evaluation_result"]["obj"])
        source_num_evals_per_category = [len(source_evaluations_per_category[i]) for i in range(len(source_evaluations_per_category))]
        source_average_evaluations_per_category = [np.mean(source_evaluations_per_category[i]) for i in range(len(source_evaluations_per_category))]
        source_average_evaluations_per_category_norm = (np.array(source_average_evaluations_per_category)-np.min(source_average_evaluations_per_category))/(np.max(source_average_evaluations_per_category)-np.min(source_average_evaluations_per_category))
        print ("source_average_evaluations_per_category: ", source_average_evaluations_per_category)

        target_evaluations_per_category = [[] for i in range(num_categories)]
        with open("gptune_tla.db/GPTUNE-TLA_HUCB_Trials"+str(trials)+"_CUCB_"+str(c_ucb)+"_CHIST_"+str(c_hist)+"-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(source_failure_handling)+"-"+str(source_n_rows)+"-"+str(source_n_cols)+"-source_mattype_"+str(source_mattype)+"-batch_num_"+str(batch_num)+".json") as f_in:
            target_function_evaluations = json.load(f_in)["func_eval"][2:]
            for i in range(len(target_function_evaluations)):
                func_eval = target_function_evaluations[i]
                wall_clock_time = np.average(func_eval["additional_output"]["parla_times"])
                normalized_residual_error_to_Axstar = np.average(func_eval["additional_output"]["normalized_residual_errors_to_Axstar"])
                if func_eval["constants"]["rls_method"] == "blendenpik" and func_eval["constants"]["sketch_operator"] == "sjlt":
                    category = 0
                elif func_eval["constants"]["rls_method"] == "lsrn" and func_eval["constants"]["sketch_operator"] == "sjlt":
                    category = 1
                elif func_eval["constants"]["rls_method"] == "newtonsketch" and func_eval["constants"]["sketch_operator"] == "sjlt":
                    category = 2
                elif func_eval["constants"]["rls_method"] == "blendenpik" and func_eval["constants"]["sketch_operator"] == "less_uniform":
                    category = 3
                elif func_eval["constants"]["rls_method"] == "lsrn" and func_eval["constants"]["sketch_operator"] == "less_uniform":
                    category = 4
                elif func_eval["constants"]["rls_method"] == "newtonsketch" and func_eval["constants"]["sketch_operator"] == "less_uniform":
                    category = 5
                target_evaluations_per_category[category].append(func_eval["evaluation_result"]["obj"])
        target_num_evals_per_category = [len(target_evaluations_per_category[i]) for i in range(len(target_evaluations_per_category))]
        print ("target_num_evals_per_category: ", target_num_evals_per_category)
        target_average_evaluations_per_category = []
        for i in range(len(target_evaluations_per_category)):
            if len(target_evaluations_per_category[i]) == 0:
                target_average_evaluations_per_category.append(0)
            else:
                target_average_evaluations_per_category.append(np.mean(target_evaluations_per_category[i]))
        #target_average_evaluations_per_category = [np.mean(target_evaluations_per_category[i]) for i in range(len(target_evaluations_per_category))]
        if np.sum(target_num_evals_per_category) == 0:
            target_average_evaluations_per_category_norm = np.zeros(num_categories)
        else:
            target_average_evaluations_per_category_norm = (np.array(target_average_evaluations_per_category)-np.min(target_average_evaluations_per_category))/(np.max(target_average_evaluations_per_category)-np.min(target_average_evaluations_per_category))
        print ("target_average_evaluations_per_category: ", target_average_evaluations_per_category)

        ucb_num_evals_per_category = c_hist * np.array(source_num_evals_per_category) + np.array(target_num_evals_per_category)
        ucb_average_evaluations_per_category = c_hist * source_average_evaluations_per_category_norm + target_average_evaluations_per_category_norm
        ucb_num_evals_total = np.sum(ucb_num_evals_per_category)
        print ("ucb_average_evaluations_per_category: ", ucb_average_evaluations_per_category)

        rewards = np.abs((np.max(ucb_average_evaluations_per_category)-ucb_average_evaluations_per_category)/(np.max(ucb_average_evaluations_per_category)-np.min(ucb_average_evaluations_per_category)))
        balance = c_ucb * np.sqrt(np.log(ucb_num_evals_total)/np.array(ucb_num_evals_per_category))
        print ("rewards: ", rewards)
        print ("balance: ", balance)
        ucb_values = rewards + balance
        best_category = np.argmax(ucb_values)
        print ("best_category: ", best_category)
        category = best_category
 
        if category == 0:
            rls_method = "blendenpik"
            sketch_operator = "sjlt"
        elif category == 1:
            rls_method = "lsrn"
            sketch_operator = "sjlt"
        elif category == 2:
            rls_method = "newtonsketch"
            sketch_operator = "sjlt"
        elif category == 3:
            rls_method = "blendenpik"
            sketch_operator = "less_uniform"
        elif category == 4:
            rls_method = "lsrn"
            sketch_operator = "less_uniform"
        elif category == 5:
            rls_method = "newtonsketch"
            sketch_operator = "less_uniform"

        """ ouput space """
        if failure_handling == "skip":
            obj = Real(float(0.0), float(900.0), name="obj", optimize=True)
        else:
            obj = Real(float("-Inf"), float("Inf"), name="obj", optimize=True)
        output_space = Space([obj])

        """ constant variables """
        constants = {
            "rls_method": rls_method,
            "sketch_operator": sketch_operator,
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

        options['sample_class'] = 'SampleLHSMDU'
        options['sample_random_seed'] = batch_num
        options['model_class'] = 'Model_GPy_LCM'
        options['model_random_seed'] = batch_num
        options['search_class'] = 'SearchPyGMO'
        options['search_random_seed'] = batch_num

        options['TLA_method'] = 'LCM'

        if failure_handling == "skip":
            options['model_output_constraint'] = 'Ignore'
        options.validate(computer=computer)

        """ run gptune """
        giventask = [[n_rows, n_cols]]
        NI=len(giventask)
        #NS=nrun

        gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb, driverabspath=os.path.abspath(__file__))

        source_function_evaluations, best_func_eval = LoadSourceFunctionEvaluations(rls_method, sketch_operator)
        #print ("source_function_evaluations: ", source_function_evaluations)

        (data, modeler, stats) = gt.TLA_I(NS=NS, Tnew=giventask, source_function_evaluations=source_function_evaluations)
        if NS == nrun:
            return (data, modeler, stats)

        NS += 1

def main():

    global A, b

    global seed

    global mattype
    global n_rows
    global n_cols

    global source_mattype
    global source_n_rows
    global source_n_cols

    global nthreads
    global x_star
    global direct_time
    global direct_relative_residual

    global batch_num

    global reference_normalized_residual_error_to_Axstar

    global failure_handling
    global source_failure_handling
    global mab_policy

    args = parse_args()
    mattype = str(args.mattype)
    print ("mattype: ", mattype)
    n_rows = args.n_rows
    print ("n_rows: ", n_rows)
    n_cols = args.n_cols
    print ("n_cols: ", n_cols)
    source_mattype = str(args.source_mattype)
    print ("source_mattype: ", source_mattype)
    source_n_rows = args.source_n_rows
    print ("source_n_rows: ", source_n_rows)
    source_n_cols = args.source_n_cols
    print ("source_n_cols: ", source_n_cols)
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
    source_failure_handling = args.source_failure_handling
    print ("source_failure_handling: ", source_failure_handling)
    mab_policy = args.mab_policy
    print ("mab_policy: ", mab_policy)

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

    if mab_policy == "source_greedy":
        (data, modeler, stats) = RunTLA_SourceGreedy(nrun)
    elif mab_policy == "source_greedy_epsilon_0.1":
        (data, modeler, stats) = RunTLA_SourceGreedyEpsilon(nrun, 0.1)
    elif mab_policy == "target_greedy_trials_1":
        (data, modeler, stats) = RunTLA_TargetGreedy(nrun, 1)
    elif mab_policy == "target_greedy_trials_1_epsilon_0.1":
        (data, modeler, stats) = RunTLA_TargetGreedyEpsilon(nrun, 1, 0.1)
    elif mab_policy == "ucb_trials_1_cucb_4":
        (data, modeler, stats) = RunTLA_UCB(nrun, 1, 4)
    elif mab_policy == "ucb_trials_0_cucb_4":
        (data, modeler, stats) = RunTLA_UCB(nrun, 0, 4)
    elif mab_policy == "hucb_trials_1_cucb_4_chist_1":
        (data, modeler, stats) = RunTLA_HUCB(nrun, 1, 4, 1)
    elif mab_policy == "hucb_trials_0_cucb_1_chist_1":
        (data, modeler, stats) = RunTLA_HUCB(nrun, 0, 1, 1)
    elif mab_policy == "hucb_trials_0_cucb_2_chist_1":
        (data, modeler, stats) = RunTLA_HUCB(nrun, 0, 2, 1)
    elif mab_policy == "hucb_trials_0_cucb_4_chist_1":
        (data, modeler, stats) = RunTLA_HUCB(nrun, 0, 4, 1)
    elif mab_policy == "hucb_trials_0_cucb_8_chist_1":
        (data, modeler, stats) = RunTLA_HUCB(nrun, 0, 8, 1)

    """ Print all input and parameter samples """
    for tid in range(NI):
        print("tid: %d" % (tid))
        print("    t: ", (data.I[tid][0]))
        print("    Ps ", data.P[tid])
        print("    Os ", data.O[tid].tolist())
        print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

if __name__ == "__main__":
    main()
