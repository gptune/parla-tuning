import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.interpolate import griddata
from matplotlib import ticker, cm
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import math

def gen_plots(n_rows, n_cols, failure_handling):

    experiment_name = "analysis_tuning_TLA_options"

    plt.rcParams["font.family"] = "Times New Roman"
    fig = plt.figure(figsize=(12,6))
    outer = gridspec.GridSpec(2, 1, wspace=0.1, hspace=0.8)

    for base in ["num_evals", "eval_time"]:
        if base == "num_evals":
            objective = "mean"
            inner = gridspec.GridSpecFromSubplotSpec(1, 4,
                            subplot_spec=outer[0], wspace=0.15, hspace=0.1)
        
            for problem_id in range(4):
                ax = plt.Subplot(fig, inner[problem_id])
        
                if problem_id == 0:
                    mattype = "GA"
                elif problem_id == 1:
                    mattype = "T5"
                elif problem_id == 2:
                    mattype = "T3"
                elif problem_id == 3:
                    mattype = "T1"
        
                #for tuner in ["lhsmdu","tpe","gptune","gptune-tla","gptune-tla1","gptune-tla2","gptune-tla3","gptune-tla4"]: #,"gptune-tla5"]:
                #for tuner in ["lhsmdu","tpe","gptune","gptune-tla","gptune-tla3"]:
                for tuner in ["gptune-tla","gptune-tla1","gptune-tla2","gptune-tla3","gptune-tla4"]:
        
                    batches_num_func_eval = []
                    batches_best_tuning_result = []
        
                    for batch_num in [1,2,3,4,5]:
                        if tuner == "lhsmdu":
                            search_logfile = "tuning_no_tla/lhsmdu.db/LHSMDU-SEARCH-n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-mattype_"+str(mattype)+"-batch_num_"+str(batch_num)+".json"
                        elif tuner == "random":
                            search_logfile = "tuning_no_tla/random_search.db/RANDOM-SEARCH-n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-mattype_"+str(mattype)+"-batch_num_"+str(batch_num)+".json"
                        elif tuner == "tpe":
                            search_logfile = "tuning_no_tla/tpe.db/TPE-SEARCH-failure_handling_"+str(failure_handling)+"-n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-mattype_"+str(mattype)+"-npilot_"+str(10)+"-batch_num_"+str(batch_num)+".json"
                        elif tuner == "gptune":
                            search_logfile = "tuning_no_tla/gptune.db/GPTUNE-SEARCH-failure_handling_"+str(failure_handling)+"-n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-mattype_"+str(mattype)+"-npilot_"+str(10)+"-batch_num_"+str(batch_num)+".json"
                        elif tuner == "gptune-tla":
                            search_logfile = "tuning_tla/gptune_tla.db/GPTUNE-TLA-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(failure_handling)+"-10000-"+str(n_cols)+"-source_mattype_"+str(mattype)+"-batch_num_"+str(batch_num)+".json"
                        elif tuner == "gptune-tla1":
                            search_logfile = "tuning_tla/gptune_tla.db/GPTUNE-TLA_HUCB_Trials0_CUCB_1_CHIST_1-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(failure_handling)+"-10000-"+str(n_cols)+"-source_mattype_"+str(mattype)+"-batch_num_"+str(batch_num)+".json"
                        elif tuner == "gptune-tla2":
                            search_logfile = "tuning_tla/gptune_tla.db/GPTUNE-TLA_HUCB_Trials0_CUCB_2_CHIST_1-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(failure_handling)+"-10000-"+str(n_cols)+"-source_mattype_"+str(mattype)+"-batch_num_"+str(batch_num)+".json"
                        elif tuner == "gptune-tla3":
                            search_logfile = "tuning_tla/gptune_tla.db/GPTUNE-TLA_HUCB_Trials0_CUCB_4_CHIST_1-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(failure_handling)+"-10000-"+str(n_cols)+"-source_mattype_"+str(mattype)+"-batch_num_"+str(batch_num)+".json"
                            search_logfile = "tuning_tla/gptune_tla.db/GPTUNE-TLA_HUCB_Trials0_CUCB_4_CHIST_1-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(failure_handling)+"-10000-"+str(n_cols)+"-source_mattype_"+str(mattype)+"-batch_num_"+str(batch_num)+".json"
                        elif tuner == "gptune-tla3":
                            search_logfile = "tuning_tla/gptune_tla.db/GPTUNE-TLA_HUCB_Trials0_CUCB_8_CHIST_1-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(failure_handling)+"-10000-"+str(n_cols)+"-source_mattype_"+str(mattype)+"-batch_num_"+str(batch_num)+".json"
        
                        print(search_logfile)
                        if not os.path.exists(search_logfile):
                            print ("file not exist")
                            continue
        
                        best_y = 100000
                        with open(search_logfile, "r") as f_in:
                            print ("tuner: ", tuner, "batch: ", batch_num, "search_logfile: ", search_logfile)
                            num_func_eval = []
                            best_tuning_result = []
                            #best_tuning_annotate = []
        
                            #if tuner == "gptune" or tuner == "random":
                            if "gptune" in tuner or "lhsmdu" in tuner:
                                function_evaluations = json.load(f_in)["func_eval"]
                            elif "random" in tuner or "tpe" in tuner:
                                function_evaluations = json.load(f_in)
        
                            reference_normalized_residual_error_to_Axstar = 0
                            for i in range(0, len(function_evaluations), 1):
                                func_eval = function_evaluations[i]
        
                                obj = func_eval["evaluation_result"]["obj"]
                                if i == 0:
                                    wall_clock_time = obj
                                    if "gptune" in tuner or "lhsmdu" in tuner:
                                        reference_normalized_residual_error_to_Axstar = np.average(func_eval["additional_output"]["normalized_residual_errors_to_Axstar"])
                                    elif "random" in tuner or "tpe" in tuner:
                                        reference_normalized_residual_error_to_Axstar = np.average(func_eval["evaluation_detail"]["normalized_residual_errors_to_Axstar"]["evaluations"])
                                else:
                                    if "gptune" in tuner or "lhsmdu" in tuner:
                                        normalized_residual_error_to_Axstar = np.average(func_eval["additional_output"]["normalized_residual_errors_to_Axstar"])
                                    elif "random" in tuner or "tpe" in tuner:
                                        normalized_residual_error_to_Axstar = np.average(func_eval["evaluation_detail"]["normalized_residual_errors_to_Axstar"]["evaluations"])
                                    if normalized_residual_error_to_Axstar > 5*(reference_normalized_residual_error_to_Axstar):
                                        wall_clock_time = 100000
                                    else:
                                        wall_clock_time = obj
        
                                if best_y > wall_clock_time:
                                    best_y = wall_clock_time
        
                                num_func_eval.append(i+1)
                                best_tuning_result.append(best_y)
                            print ("num_func_eval: ", num_func_eval)
                            print ("best_tuning_result: ", best_tuning_result)
        
                            #point_list = [i for i in range(1, num_function_evaluations+1, 1)]
                            point_list = [i for i in range(1, len(num_func_eval)+1, 1)]
                            num_func_eval = [num_func_eval[i-1] for i in point_list]
                            best_tuning_result = [best_tuning_result[i-1] for i in point_list]
                            print ("num_func_eval: ", num_func_eval)
                            print ("best_tuning_result: ", best_tuning_result)
                            batches_num_func_eval.append(num_func_eval)
                            batches_best_tuning_result.append(best_tuning_result)

                    if tuner == "lhsmdu":
                        label_name = "LHSMDU"
                        color_code = "tab:blue"
                    elif tuner == "tpe":
                        label_name = "TPE"
                        color_code = "tab:orange"
                    elif tuner == "gptune":
                        label_name = "GPTune"
                        color_code = "tab:green"
                    elif tuner == "gptune-tla":
                        label_name = "Original"
                        color_code = "tab:gray"
                    elif tuner == "gptune-tla1":
                        label_name = "HUCB (c=1)"
                        color_code = "tab:pink"
                    elif tuner == "gptune-tla2":
                        label_name = "HUCB (c=2)"
                        color_code = "tab:cyan"
                    elif tuner == "gptune-tla3":
                        label_name = "HUCB (c=4)"
                        #label_name = "GPTune (transfer learning)"
                        color_code = "tab:red"
                    elif tuner == "gptune-tla4":
                        label_name = "HUCB (c=8)"
                        color_code = "tab:purple"

                    # plotting
                    if objective == "median" or objective == "mean":
                        num_func_eval = batches_num_func_eval[0]
                        if objective == "median":
                            best_tuning_result = np.median(batches_best_tuning_result, axis=0)
                        elif objective == "mean":
                            best_tuning_result = np.mean(batches_best_tuning_result, axis=0)
                        print ("tuner: " ,tuner)
                        print ("num_func_eval: ", num_func_eval)
                        print ("best_tuning_result: ", best_tuning_result)
                        solution = round(best_tuning_result[-1],3)
                        print ("solution: ", solution)
                        best_tuning_result_lower = np.std(batches_best_tuning_result, axis=0)
                        best_tuning_result_upper = np.std(batches_best_tuning_result, axis=0)
                        print ("npstd: ", np.std(batches_best_tuning_result, axis=0))
        
                        ax.plot(num_func_eval, best_tuning_result, label=label_name, linewidth=2.5, color=color_code)
                        ax.fill_between(num_func_eval, best_tuning_result-np.std(batches_best_tuning_result, axis=0), best_tuning_result+np.std(batches_best_tuning_result, axis=0), color=color_code, alpha=0.2)
        
                    elif objective == "std":
                        num_func_eval = batches_num_func_eval[0]
                        tuning_result_std = np.std(batches_best_tuning_result, axis=0)
                        ax.plot(num_func_eval, tuning_result_std, label=label_name, linewidth=2.5, color=color_code)
        
                if objective == "median" or objective == "mean":
                    ax.set_title("Matrix: "+mattype)

                    if problem_id == 0:
                        ax.legend(loc='upper right', ncol=1)

                    ax.set_xlim(1, 50)
                    ax.set_xticks([1,10,20,30,40,50])
                    ax.set_xticklabels(["3","10","20","30","40","50"])

                    if mattype == "T1":
                        ax.set_ylim(0.8, 2.0)
                        ax.set_yticks([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
                        ax.set_yticklabels(["0.8","1.0","1.2","1.4","1.6","1.8","2.0"])
                    else:
                        ax.set_ylim(0.6, 1.4)

                    ax.set_xlabel("Number of function evaluations", fontsize=12)
                    if problem_id == 0:
                        ax.set_ylabel("Tuned performance \n (wall-clock time (s))", fontsize=12)
        
                    ax.yaxis.grid()
        
                elif objective == "std":
                    ax.set_title("Matrix: "+mattype)

                    ax.set_xlim(1, 50)
                    ax.set_xticks([1,10,20,30,40,50])
                    ax.set_xticklabels(["3","10","20","30","40","50"])
        
                    ax.set_xlabel("Number of function evaluations", fontsize=12)
                    if problem_id == 0:
                        ax.set_ylabel("Standard deviation of \n tuned performance", fontsize=12)
        
                    ax.yaxis.grid()
        
                fig.add_subplot(ax)

        elif base == "eval_time":
            objective = "mean"
            inner = gridspec.GridSpecFromSubplotSpec(1, 4,
                            subplot_spec=outer[1], wspace=0.15, hspace=0.1)
    
            for problem_id in range(4):
                ax = plt.Subplot(fig, inner[problem_id])
    
                if problem_id == 0:
                    mattype = "GA"
                elif problem_id == 1:
                    mattype = "T5"
                elif problem_id == 2:
                    mattype = "T3"
                elif problem_id == 3:
                    mattype = "T1"
    
                #for tuner in ["lhsmdu","tpe","gptune","gptune-tla","gptune-tla1","gptune-tla2","gptune-tla3","gptune-tla4"]: #,"gptune-tla5"]:
                #for tuner in ["lhsmdu","tpe","gptune","gptune-tla","gptune-tla3"]:
                for tuner in ["gptune-tla","gptune-tla1","gptune-tla2","gptune-tla3","gptune-tla4"]:
    
                    batches_num_func_eval = []
                    batches_best_tuning_result = []
    
                    for batch_num in [1,2,3,4,5]:
                        if tuner == "lhsmdu":
                            search_logfile = "tuning_no_tla/lhsmdu.db/LHSMDU-SEARCH-n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-mattype_"+str(mattype)+"-batch_num_"+str(batch_num)+".json"
                        elif tuner == "random":
                            search_logfile = "tuning_no_tla/random_search.db/RANDOM-SEARCH-n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-mattype_"+str(mattype)+"-batch_num_"+str(batch_num)+".json"
                        elif tuner == "tpe":
                            search_logfile = "tuning_no_tla/tpe.db/TPE-SEARCH-failure_handling_"+str(failure_handling)+"-n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-mattype_"+str(mattype)+"-npilot_"+str(10)+"-batch_num_"+str(batch_num)+".json"
                        elif tuner == "gptune":
                            search_logfile = "tuning_no_tla/gptune.db/GPTUNE-SEARCH-failure_handling_"+str(failure_handling)+"-n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-mattype_"+str(mattype)+"-npilot_"+str(10)+"-batch_num_"+str(batch_num)+".json"
                        elif tuner == "gptune-tla":
                            search_logfile = "tuning_tla/gptune_tla.db/GPTUNE-TLA-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(failure_handling)+"-10000-"+str(n_cols)+"-source_mattype_"+str(mattype)+"-batch_num_"+str(batch_num)+".json"
                        elif tuner == "gptune-tla1":
                            search_logfile = "tuning_tla/gptune_tla.db/GPTUNE-TLA_HUCB_Trials0_CUCB_1_CHIST_1-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(failure_handling)+"-10000-"+str(n_cols)+"-source_mattype_"+str(mattype)+"-batch_num_"+str(batch_num)+".json"
                        elif tuner == "gptune-tla2":
                            search_logfile = "tuning_tla/gptune_tla.db/GPTUNE-TLA_HUCB_Trials0_CUCB_2_CHIST_1-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(failure_handling)+"-10000-"+str(n_cols)+"-source_mattype_"+str(mattype)+"-batch_num_"+str(batch_num)+".json"
                        elif tuner == "gptune-tla3":
                            search_logfile = "tuning_tla/gptune_tla.db/GPTUNE-TLA_HUCB_Trials0_CUCB_4_CHIST_1-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(failure_handling)+"-10000-"+str(n_cols)+"-source_mattype_"+str(mattype)+"-batch_num_"+str(batch_num)+".json"
                            search_logfile = "tuning_tla/gptune_tla.db/GPTUNE-TLA_HUCB_Trials0_CUCB_4_CHIST_1-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(failure_handling)+"-10000-"+str(n_cols)+"-source_mattype_"+str(mattype)+"-batch_num_"+str(batch_num)+".json"
                        elif tuner == "gptune-tla4":
                            search_logfile = "tuning_tla/gptune_tla.db/GPTUNE-TLA_HUCB_Trials0_CUCB_8_CHIST_1-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(failure_handling)+"-10000-"+str(n_cols)+"-source_mattype_"+str(mattype)+"-batch_num_"+str(batch_num)+".json"
    
                        print(search_logfile)
                        if not os.path.exists(search_logfile):
                            print ("file not exist")
                            continue
    
                        best_y = 100000
                        with open(search_logfile, "r") as f_in:
                            print ("tuner: ", tuner, "batch: ", batch_num, "search_logfile: ", search_logfile)
    
                            total_evaluation_time = []
                            best_tuning_result = []
    
                            accumulated_evaluation_time = 0
                            accumulated_evaluation_time_s = 0
    
                            if "gptune" in tuner or "lhsmdu" in tuner:
                                function_evaluations = json.load(f_in)["func_eval"]
                            elif "random" in tuner or "tpe" in tuner:
                                function_evaluations = json.load(f_in)
    
                            prior_result = -1
    
                            reference_normalized_residual_error_to_Axstar = 0
                            for i in range(0, len(function_evaluations), 1):
                                func_eval = function_evaluations[i]
    
                                obj = func_eval["evaluation_result"]["obj"]
    
                                if i == 0:
                                    wall_clock_time = obj
                                    if "gptune" in tuner or "lhsmdu" in tuner:
                                        reference_normalized_residual_error_to_Axstar = np.average(func_eval["additional_output"]["normalized_residual_errors_to_Axstar"])
                                    elif "random" in tuner or "tpe" in tuner:
                                        reference_normalized_residual_error_to_Axstar = np.average(func_eval["evaluation_detail"]["normalized_residual_errors_to_Axstar"]["evaluations"])
                                else:
                                    if "gptune" in tuner or "lhsmdu" in tuner:
                                        normalized_residual_error_to_Axstar = np.average(func_eval["additional_output"]["normalized_residual_errors_to_Axstar"])
                                    elif "random" in tuner or "tpe" in tuner:
                                        normalized_residual_error_to_Axstar = np.average(func_eval["evaluation_detail"]["normalized_residual_errors_to_Axstar"]["evaluations"])
                                    if normalized_residual_error_to_Axstar > 5*(reference_normalized_residual_error_to_Axstar):
                                        wall_clock_time = 100000
                                    else:
                                        wall_clock_time = obj
    
                                if best_y > wall_clock_time:
                                    best_y = wall_clock_time
    
                                if "gptune" in tuner or "lhsmdu" in tuner:
                                    evaluation_time = np.sum(func_eval["additional_output"]["parla_times"])
                                elif "random" in tuner or "tpe" in tuner:
                                    evaluation_time = np.sum(func_eval["evaluation_detail"]["wall_clock_time"]["evaluations"])
    
                                accumulated_evaluation_time += evaluation_time
                                while accumulated_evaluation_time_s < accumulated_evaluation_time:
                                    best_tuning_result.append(prior_result)
                                    accumulated_evaluation_time_s += 1
    
                                prior_result = best_y
    
                            print ("best_tuning_result: ", best_tuning_result)
                            print ("num_elements: ", len(best_tuning_result))
                            batches_best_tuning_result.append(best_tuning_result)

                    if tuner == "lhsmdu":
                        label_name = "LHSMDU"
                        color_code = "tab:blue"
                    elif tuner == "tpe":
                        label_name = "TPE"
                        color_code = "tab:orange"
                    elif tuner == "gptune":
                        label_name = "GPTune"
                        color_code = "tab:green"
                    elif tuner == "gptune-tla":
                        label_name = "Original"
                        color_code = "tab:gray"
                    elif tuner == "gptune-tla1":
                        label_name = "HUCB (c=1)"
                        color_code = "tab:pink"
                    elif tuner == "gptune-tla2":
                        label_name = "HUCB (c=2)"
                        color_code = "tab:cyan"
                    elif tuner == "gptune-tla3":
                        label_name = "HUCB (c=4)"
                        #label_name = "GPTune (transfer learning)"
                        color_code = "tab:red"
                    elif tuner == "gptune-tla4":
                        label_name = "HUCB (c=8)"
                        color_code = "tab:purple"
   
                    # plotting
                    if objective == "median" or objective == "mean":
                        min_bound = min(len(batches_best_tuning_result[i]) for i in range(len(batches_best_tuning_result)))
    
                        start_point = 0
                        for i in range(min_bound):
                            not_yet = False
                            for j in range(len(batches_best_tuning_result)):
                                if batches_best_tuning_result[j][i] == -1:
                                    not_yet = True
                            if not_yet == True:
                                start_point += 1
                        num_func_eval = list(np.arange(start_point,min_bound,1))
                        batches_best_tuning_result_ = [batches_best_tuning_result[i][start_point:min_bound] for i in range(len(batches_best_tuning_result))]
    
                        print ("batches_best_tuning_result_: ", batches_best_tuning_result_)
    
                        if objective == "median":
                            best_tuning_result = np.median(batches_best_tuning_result_, axis=0)
                        elif objective == "mean":
                            best_tuning_result = np.mean(batches_best_tuning_result_, axis=0)
    
                        print ("tuner: " ,tuner)
                        print ("num_func_eval: ", num_func_eval)
                        print ("best_tuning_result: ", best_tuning_result)
    
                        ax.plot(num_func_eval, best_tuning_result, label=label_name, linewidth=2.5, color=color_code)
                        ax.fill_between(num_func_eval, best_tuning_result-np.std(batches_best_tuning_result_, axis=0), best_tuning_result+np.std(batches_best_tuning_result_, axis=0), color=color_code, alpha=0.2)
    
                    elif objective == "std":
                        min_bound = min(len(batches_best_tuning_result[i]) for i in range(len(batches_best_tuning_result)))
                        start_point = 0
                        for i in range(min_bound):
                            not_yet = False
                            for j in range(len(batches_best_tuning_result)):
                                if batches_best_tuning_result[j][i] == -1:
                                    not_yet = True
                            if not_yet == True:
                                start_point += 1
                        num_func_eval = list(np.arange(start_point,min_bound,1))
                        batches_best_tuning_result_ = [batches_best_tuning_result[i][start_point:min_bound] for i in range(len(batches_best_tuning_result))]
                        tuning_result_std = np.std(batches_best_tuning_result_, axis=0)
    
                        ax.plot(num_func_eval, tuning_result_std, label=label_name, linewidth=2.5, color=color_code)
    
                if objective == "median" or objective == "mean":
                    ax.set_title("Matrix: "+mattype)
    
                    if mattype == "GA" or mattype =="T5":
                        ax.set_xlim(0, 600)
                        ax.set_xticks([0,200,400,600,800])
                        ax.set_xticklabels(["0","200","400","600","800"])
                    elif mattype == "T3":
                        ax.set_xlim(0, 800)
                        ax.set_xticks([0,200,400,600,800])
                        ax.set_xticklabels(["0","200","400","600","800"])
                    elif mattype == "T1":
                        ax.set_xlim(0, 1500)
                        ax.set_xticks([0,500,1000,1500])
                        ax.set_xticklabels(["0","500","1000","1500"])
                    #ax.set_xlim(0, 450)

                    if mattype == "T1":
                        ax.set_ylim(0.8, 2.0)
                        ax.set_yticks([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
                        ax.set_yticklabels(["0.8","1.0","1.2","1.4","1.6","1.8","2.0"])
                    else:
                        ax.set_ylim(0.6, 1.4)

                    ax.set_xlabel("Function evaluation time (s)", fontsize=12)
                    if problem_id == 0:
                        ax.set_ylabel("Tuned performance \n (wall-clock time (s))", fontsize=12)
    
                    ax.yaxis.grid()
    
                elif objective == "std":
                    ax.set_title("Matrix: "+mattype)
    
                    if mattype == "GA" or mattype =="T5":
                        ax.set_xlim(0, 1000)
                        ax.set_xticks([0,200,400,600,800,1000])
                        ax.set_xticklabels(["0","200","400","600","800","1000"])
                    elif mattype == "T3":
                        ax.set_xlim(0, 1200)
                        ax.set_xticks([0,200,400,600,800,1000,1200])
                        ax.set_xticklabels(["0","200","400","600","800","1000","1200"])
                    elif mattype == "T1":
                        ax.set_xlim(0, 2000)
                        ax.set_xticks([0,500,1000,1500,2000])
                        ax.set_xticklabels(["0","500","1000","1500","2000"])
                    #ax.set_xlim(3, 50)
    
                    ax.set_xlabel("Function evaluation time (s)", fontsize=12)
                    if problem_id == 0:
                        ax.set_ylabel("Standard deviation of \n tuned performance", fontsize=12)
    
                    ax.yaxis.grid()
    
                fig.add_subplot(ax)

    fig_title = "Tuning of the SAP algorithms (m: "+str(n_rows)+", n: "+str(n_cols)+")"
    fig.suptitle(fig_title, fontsize=15)

    fig.text(0.25, 0.50,
            "(a) Tuned performance depending on the number of function evaluations",
            fontsize = 16,
            color = "black")

    fig.text(0.23, 0.03,
            "(b) Tuned performance depending on the accumulated function evaluation time",
            fontsize = 16,
            color = "black")

    fig.subplots_adjust(top=0.88, bottom=0.15, left=0.06, right=0.98, wspace=0.02, hspace=0.02)
    fig.savefig("plots/"+experiment_name+".pdf")

if __name__ == "__main__":

    if not os.path.exists("plots"):
        os.system("mkdir -p plots")

    n_rows = 50000
    n_cols = 1000
    gen_plots(n_rows, n_cols, "highval")

