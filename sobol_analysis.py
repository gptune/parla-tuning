#! /usr/bin/env python

# GPTune Copyright (c) 2019, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S.Dept. of Energy) and the University of
# California, Berkeley.  All rights reserved.
#
# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
#
# NOTICE. This Software was developed under funding from the U.S. Department
# of Energy and the U.S. Government consequently retains certain rights.
# As such, the U.S. Government has been granted for itself and others acting
# on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in
# the Software to reproduce, distribute copies to the public, prepare
# derivative works, and perform publicly and display publicly, and to permit
# other to do so.
#

import json
import os
import sys
sys.path.insert(0, os.path.abspath(__file__ + "/../../GPTune/GPTune/"))

global analysis_dir
global analysis_output

def sobol_analysis(n_rows, n_cols, mattype, failure_handling):

    global analysis_dir
    global analysis_output

    input_dir = "tuning_tla/lhsmdu.db/"
    input_file = "LHSMDU-SEARCH-failure_handling_"+str(failure_handling)+"-n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-mattype_"+str(mattype)+"-batch_num_1.json"

    #input_dir = "tuning_tla_mab_mil_update/lhsmdu.db/"
    #input_file = "LHSMDU-SEARCH-failure_handling_"+str(failure_handling)+"-n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-mattype_"+str(mattype)+"-batch_num_1.json"

    print ("input_file: ", input_file)

    function_evaluations = []
    with open(input_dir + "/" + input_file, "r") as f_in:
        func_eval = json.load(f_in)["func_eval"]
        for each in func_eval:
            already_exist = False
            for exist in function_evaluations:
                if each["tuning_parameter"]["rls_method"] == exist["tuning_parameter"]["rls_method"] and\
                   each["tuning_parameter"]["sketch_operator"] == exist["tuning_parameter"]["sketch_operator"] and\
                   each["tuning_parameter"]["sampling_factor"] == exist["tuning_parameter"]["sampling_factor"] and\
                   each["tuning_parameter"]["vec_nnz"] == exist["tuning_parameter"]["vec_nnz"] and\
                   each["tuning_parameter"]["safety_exponent"] == exist["tuning_parameter"]["safety_exponent"]:
                    already_exist = True
            if already_exist == False:
                function_evaluations.append(each)
    print ("loaded function evaluations: ", len(function_evaluations))

    problem_space = {
        "input_space": [
            {"name":"m", "type":"integer", "transformer":"normalize", "lower_bound":1000, "upper_bound":100000},
            {"name":"n", "type":"integer", "transformer":"normalize", "lower_bound":1000, "upper_bound":10000}
        ],
        "parameter_space": [
            {"name":"rls_method", "type":"categorical", "transformer":"onehot", "categories":['blendenpik','lsrn','newtonsketch']},
            {"name":"sketch_operator", "type":"categorical", "transformer":"onehot", "categories":["sjlt","less_uniform"]},
            {"name":"sampling_factor", "type":"real", "transformer":"normalize", "lower_bound":1.0, "upper_bound":10.0},
            {"name":"vec_nnz", "type":"integer", "transformer":"normalize", "lower_bound":1, "upper_bound":100},
            {"name":"safety_exponent", "type":"integer", "transformer":"normalize", "lower_bound":0, "upper_bound":5.0}
        ],
        "output_space": [
            {"name":"obj", "type":"real", "transformer":"identity", "lower_bound":float("-Inf"), "upper_bound":float("inf")}
        ]
    }

    import gptune
    ret = gptune.SensitivityAnalysis(
    problem_space = problem_space,
              modeler = modeling,
              method = "Sobol",
              input_task = [n_rows,n_cols],
              function_evaluations = function_evaluations,
              num_samples = n_samples
          )
    print (ret)

    if not os.path.exists(analysis_dir+"/"+analysis_output+".json"):
        json_data = []
    else:
        with open(analysis_dir+"/"+analysis_output+".json", "r") as f_in:
            json_data = json.load(f_in)

    with open(analysis_dir+"/"+analysis_output+".json", "w") as f_out:
        ret["input_problem"] = {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "mattype": mattype,
            "input_file": input_file,
            "num_function_evaluations": len(function_evaluations),
            "num_samples_for_sobol_analysis": n_samples
        }
        json_data.append(ret)
        json.dump(json_data, f_out, indent=2)
    with open(analysis_dir+"/"+analysis_output+".csv", "a") as f_out:
        f_out.write("n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-mattype_"+str(mattype)+", ")
        f_out.write("rls_method: "+str(round(ret["S1"]["rls_method"],2)) + " & sketch_operator: "+str(round(ret["S1"]["rls_method"],2)) + " & sampling_factor: "+ str(round(ret["S1"]["sampling_factor"],2)) + " & vec_nnz: "+str(round(ret["S1"]["vec_nnz"],2))+", ")
        f_out.write("rls_method: "+str(round(ret["S1_conf"]["rls_method"],2)) + " & sketch_operator: "+str(round(ret["S1_conf"]["rls_method"],2)) + " & sampling_factor: "+ str(round(ret["S1_conf"]["sampling_factor"],2)) + " & vec_nnz: "+str(round(ret["S1_conf"]["vec_nnz"],2))+", ")
        f_out.write("rls_method: "+str(round(ret["ST"]["rls_method"],2)) + " & sketch_operator: "+str(round(ret["ST"]["rls_method"],2)) + " & sampling_factor: "+ str(round(ret["ST"]["sampling_factor"],2)) + " & vec_nnz: "+str(round(ret["ST"]["vec_nnz"],2))+", ")
        f_out.write("rls_method: "+str(round(ret["ST_conf"]["rls_method"],2)) + " & sketch_operator: "+str(round(ret["ST_conf"]["rls_method"],2)) + " & sampling_factor: "+ str(round(ret["ST_conf"]["sampling_factor"],2)) + " & vec_nnz: "+str(round(ret["ST_conf"]["vec_nnz"],2))+"\n")
    with open(analysis_dir+"/"+analysis_output+".tex", "a") as f_out:
        #f_out.write("n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-condnum_"+str(condnum)+"-coherence_type_"+str(coherence_type)+"-tolerance_"+str(tolerance)+" & ")
        f_out.write(str(mattype) + " & ")
        f_out.write(str(round(ret["S1"]["rls_method"],2)) + " (" + str(round(ret["S1_conf"]["rls_method"],2)) + ") & ")
        f_out.write(str(round(ret["S1"]["sketch_operator"],2)) + " (" + str(round(ret["S1_conf"]["sketch_operator"],2)) + ") & ")
        f_out.write(str(round(ret["S1"]["sampling_factor"],2)) + " (" + str(round(ret["S1_conf"]["sampling_factor"],2)) + ") & ")
        f_out.write(str(round(ret["S1"]["vec_nnz"],2)) + " (" + str(round(ret["S1_conf"]["vec_nnz"],2)) + ") & ")
        f_out.write(str(round(ret["S1"]["safety_exponent"],2)) + " (" + str(round(ret["S1_conf"]["safety_exponent"],2)) + ") & ")
        f_out.write(str(round(ret["ST"]["rls_method"],2)) + " (" + str(round(ret["ST_conf"]["rls_method"],2)) + ") & ")
        f_out.write(str(round(ret["ST"]["sketch_operator"],2)) + " (" + str(round(ret["ST_conf"]["sketch_operator"],2)) + ") & ")
        f_out.write(str(round(ret["ST"]["sampling_factor"],2)) + " (" + str(round(ret["ST_conf"]["sampling_factor"],2)) + ") & ")
        f_out.write(str(round(ret["ST"]["vec_nnz"],2)) + " (" + str(round(ret["ST_conf"]["vec_nnz"],2)) + ") & ")
        f_out.write(str(round(ret["ST"]["safety_exponent"],2)) + " (" + str(round(ret["ST_conf"]["safety_exponent"],2)) + ")\n")

    return

if __name__ == "__main__":

    if not os.path.exists("sobol_analysis"):
        os.system("mkdir -p sobol_analysis")

    global analysis_dir
    global analysis_output

    n_samples = 512
    modeling = "Model_GPy_LCM"
    failure_handling = "highval"

    analysis_dir = "sobol_analysis"
    analysis_output = "sobol_analysis-failure_handling_"+str(failure_handling)+"-modeler_"+str(modeling)+"-n_samples_"+str(n_samples)

    with open(analysis_dir+"/"+analysis_output+".csv","a") as f_out:
        f_out.write("input problem, S1, S1_conf, ST, ST_conf\n")

    n_rows = 10000
    n_cols = 1000
    tolerance = 1e-6
    for mattype in ["GA", "T5", "T3", "T1"]:
        sobol_analysis(n_rows, n_cols, mattype, failure_handling)

