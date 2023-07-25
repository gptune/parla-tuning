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

def query_best_result(n_rows, n_cols, failure_handling, mattype):
    tolerance = 1e-6
    dbfile = "GRID-SEARCH-n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-mattype_"+str(mattype)+"-tolerance_"+str(tolerance)+".json"
    with open("grid_search/grid_search.db/"+dbfile, "r") as f_in:
        function_evaluations = json.load(f_in)
        for func_eval in function_evaluations:
            if func_eval["tuning_parameter"]["rls_method"] == "blendenpik" and \
               func_eval["tuning_parameter"]["sketch_operator"] == "sjlt" and \
               func_eval["tuning_parameter"]["sampling_factor"] == 5.0 and \
               func_eval["tuning_parameter"]["vec_nnz"] == 50:
                reference_normalized_residual_error_to_Axstar = func_eval["evaluation_result"]["normalized_residual_error_to_Axstar"]

    num_evals = 0
    total_evaluation_time = 0

    best_wall_clock_time = -1
    for tolerance in [1e-6, 1e-8, 1e-10]:
        if tolerance == 1e-6:
            tolerance_level = 0
        elif tolerance == 1e-8:
            tolerance_level = 1
        elif tolerance == 1e-10:
            tolerance_level = 2

        dbfile = "GRID-SEARCH-n_rows_"+str(n_rows)+"-n_cols_"+str(n_cols)+"-mattype_"+str(mattype)+"-tolerance_"+str(tolerance)+".json"
        with open("grid_search/grid_search.db/"+dbfile, "r") as f_in:
            function_evaluations = json.load(f_in)

        for func_eval in function_evaluations:
            rls_method = func_eval["tuning_parameter"]["rls_method"]
            sketch_operator = func_eval["tuning_parameter"]["sketch_operator"]
            sampling_factor = func_eval["tuning_parameter"]["sampling_factor"]
            vec_nnz = func_eval["tuning_parameter"]["vec_nnz"]
            normalized_residual_error_to_Axstar = func_eval["evaluation_result"]["normalized_residual_error_to_Axstar"]
            wall_clock_time = func_eval["evaluation_result"]["wall_clock_time"]

            total_evaluation_time += np.sum(func_eval["evaluation_detail"]["wall_clock_time"]["evaluations"])

            if rls_method == "blendenpik" and sketch_operator == "sjlt":
                category = 0
            elif rls_method == "lsrn" and sketch_operator == "sjlt":
                category = 1
            elif rls_method == "newtonsketch" and sketch_operator == "sjlt":
                category = 2
            elif rls_method == "blendenpik" and sketch_operator == "less_uniform":
                category = 3
            elif rls_method == "lsrn" and sketch_operator == "less_uniform":
                category = 4
            elif rls_method == "newtonsketch" and sketch_operator == "less_uniform":
                category = 5

            sampling_factor_map = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9}
            vec_nnz_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 20: 10, 30: 11, 40: 12, 50: 13, 60: 14, 70: 15, 80: 16, 90: 17, 100: 18}
            if normalized_residual_error_to_Axstar <= 10*(reference_normalized_residual_error_to_Axstar):
                if best_wall_clock_time == -1 or wall_clock_time < best_wall_clock_time:
                    best_wall_clock_time = wall_clock_time

            num_evals += 1

    return best_wall_clock_time, num_evals, total_evaluation_time

def gen_plots(n_rows, n_cols, failure_handling):

    experiment_name = "analysis_tuning_mean_source_matrices"

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

                best_result_by_grid_search, num_total_grid_evaluations, num_total_grid_evaluation_time = query_best_result(n_rows, n_cols, failure_handling, mattype)
                ax.axhline(best_result_by_grid_search, color="black", linestyle="-", label="Peak perf.")
                #label = str(round(float(best_result_by_grid_search), 2))+"s (from " + str(num_total_grid_evaluations) + " grid evaluations)"
                #label = str(num_total_grid_evaluations)+" evaluations"
                #ax.annotate(label, # this is the text
                #        (5,best_result_by_grid_search), # this is the point to label
                #        color="black",
                #        textcoords="offset points", # how to position the text
                #        fontsize=12,
                #        weight="bold",
                #        xytext=(30,3), # distance from text to points (x,y)
                #        ha='center') # horizontal alignment can be left, right or center

                #for tuner in ["lhsmdu","tpe","gptune","gptune-tla","gptune-tla1","gptune-tla2","gptune-tla3","gptune-tla4"]: #,"gptune-tla5"]:
                #for tuner in ["lhsmdu","tpe","gptune","gptune-tla","gptune-tla3"]:
                #for tuner in ["gptune-tla1"]:
                tuner = "gptune-tla3"
                for source_mat in ["T1", "T3", "T5", "GA"]:
                    batches_num_func_eval = []
                    batches_best_tuning_result = []
        
                    for batch_num in [1,2,3,4,5]:
                        search_logfile = "tuning_tla/gptune_tla.db/GPTUNE-TLA_HUCB_Trials0_CUCB_4_CHIST_1-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(failure_handling)+"-10000-"+str(n_cols)+"-source_mattype_"+str(source_mat)+"-batch_num_"+str(batch_num)+".json"
        
                        print(search_logfile)
                        if not os.path.exists(search_logfile):
                            print ("file not exist")
                            continue
        
                        best_y = 100000
                        with open(search_logfile, "r") as f_in:
                            #print ("tuner: ", tuner, "batch: ", batch_num, "search_logfile: ", search_logfile)
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
        
                    label_name = "Source matrix: "+str(source_mat)
                    if source_mat == "T1":
                        color_code = "tab:orange"
                    elif source_mat == "T3":
                        color_code = "tab:pink"
                    elif source_mat == "T5":
                        color_code = "tab:purple"
                    elif source_mat == "GA":
                        color_code = "tab:gray"
        
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
                        ax.legend(loc='upper right')

                    ax.set_xlim(1, 50)
                    ax.set_xticks([1,10,20,30,40,50])
                    ax.set_xticklabels(["3","10","20","30","40","50"])
        
                    ax.set_ylim(0.5, 2.0)
        
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

                best_result_by_grid_search, num_total_grid_evaluations, num_total_grid_evaluation_time = query_best_result(n_rows, n_cols, failure_handling, mattype)
                ax.axhline(best_result_by_grid_search, color="black", linestyle="-", label="Peak perf.")
                #label = str(round(float(best_result_by_grid_search), 2))+"s (from " + str(num_total_grid_evaluation_time) + " grid evaluation time)"
                #label = str(round(num_total_grid_evaluation_time,2)) + "s evaluation time"
                #ax.annotate(label, # this is the text
                #        (5,best_result_by_grid_search), # this is the point to label
                #        color="black",
                #        textcoords="offset points", # how to position the text
                #        fontsize=10,
                #        weight="bold",
                #        xytext=(80,-10), # distance from text to points (x,y)
                #        ha='center') # horizontal alignment can be left, right or center
        
                #for tuner in ["lhsmdu","tpe","gptune","gptune-tla","gptune-tla1","gptune-tla2","gptune-tla3","gptune-tla4"]: #,"gptune-tla5"]:
                #for tuner in ["lhsmdu","tpe","gptune","gptune-tla","gptune-tla3"]:
                tuner = "gptune-tla3"
                for source_mat in ["T1", "T3", "T5", "GA"]:
                    batches_num_func_eval = []
                    batches_num_func_eval = []
                    batches_best_tuning_result = []
    
                    for batch_num in [1,2,3,4,5]:
                        search_logfile = "tuning_tla/gptune_tla.db/GPTUNE-TLA_HUCB_Trials0_CUCB_4_CHIST_1-target-failure_handling_"+str(failure_handling)+"-"+str(n_rows)+"-"+str(n_cols)+"-mattype_"+str(mattype)+"-source-failure_handling_"+str(failure_handling)+"-10000-"+str(n_cols)+"-source_mattype_"+str(source_mat)+"-batch_num_"+str(batch_num)+".json"
    
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
    
                    label_name = "Source matrix: "+str(source_mat)
                    if source_mat == "T1":
                        color_code = "tab:orange"
                    elif source_mat == "T3":
                        color_code = "tab:pink"
                    elif source_mat == "T5":
                        color_code = "tab:purple"
                    elif source_mat == "GA":
                        color_code = "tab:gray"
    
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
                        #ax.set_xlim(0, 400) #1000)
                        #ax.set_xticks([0,200,400]) #,600,800,1000])
                        #ax.set_xticklabels(["0","200","400"]) #,"600","800","1000"])
                        ax.set_xlim(0, 600)
                        ax.set_xticks([0,200,400,600])
                        ax.set_xticklabels(["0","200","400","600"])
                    elif mattype == "T3":
                        #ax.set_xlim(0, 400) #1000)
                        #ax.set_xticks([0,200,400]) #,600,800,1000])
                        #ax.set_xticklabels(["0","200","400"]) #,"600","800","1000"])
                        ax.set_xlim(0, 80)
                        ax.set_xticks([0,200,400,600,800])
                        ax.set_xticklabels(["0","200","400","600","800"])
                    elif mattype == "T1":
                        #ax.set_xlim(0, 400) #1000)
                        #ax.set_xticks([0,200,400]) #,600,800,1000])
                        #ax.set_xticklabels(["0","200","400"]) #,"600","800","1000"])
                        ax.set_xlim(0, 1200)
                        ax.set_xticks([0,200,400,600,800,1000,1200])
                        ax.set_xticklabels(["0","200","400","600","800","1000","1200"])
                    #ax.set_xlim(0, 450)
                    ax.set_ylim(0.5, 2.0)
    
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

