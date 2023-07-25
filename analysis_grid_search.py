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
from matplotlib.colors import LinearSegmentedColormap
import os
import math

def gen_plot():

    n_rows = 50000
    n_cols = 1000
    failure_handling = "highval"

    experiment_name = "analysis_grid_search"

    plt.rcParams["font.family"] = "Times New Roman"
    fig = plt.figure(figsize=(12,6))
    outer = gridspec.GridSpec(1, 4, wspace=0.1, hspace=0.2)

    z_min = 0.5
    z_max = 40

    for matrix_id in range(4):
        if matrix_id == 0:
            mattype = "GA"
        elif matrix_id == 1:
            mattype = "T5"
        elif matrix_id == 2:
            mattype = "T3"
        elif matrix_id == 3:
            mattype = "T1"

        num_tolerance_levels = 3
        num_sampling_factor_levels = 10
        num_vec_nnz_levels = 19
        num_categories = 6

        wall_clock_times = [[[[-1 for l in range(num_tolerance_levels)] for k in range(num_vec_nnz_levels) ] for j in range(num_sampling_factor_levels)] for i in range(num_categories)]

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
                    wall_clock_times[category][sampling_factor_map[sampling_factor]][vec_nnz_map[vec_nnz]][tolerance_level] = wall_clock_time

        inner = gridspec.GridSpecFromSubplotSpec(2, 3,
                        subplot_spec=outer[matrix_id], wspace=0.05, hspace=0.02)

        for category_id in range(6):
            ax = plt.Subplot(fig, inner[category_id])
            if category_id == 0:
                rls_method = "blendenpik"
                sketch_operator = "sjlt"
            elif category_id == 1:
                rls_method = "lsrn"
                sketch_operator = "sjlt"
            elif category_id == 2:
                rls_method = "newtonsketch"
                sketch_operator = "sjlt"
            elif category_id == 3:
                rls_method = "blendenpik"
                sketch_operator = "less_uniform"
            elif category_id == 4:
                rls_method = "lsrn"
                sketch_operator = "less_uniform"
            elif category_id == 5:
                rls_method = "newtonsketch"
                sketch_operator = "less_uniform"

            x_ = []
            y_ = []
            z_ = []

            best_x, best_y, best_wall_clock_time_result = -1, -1, -1
            for j in range(num_sampling_factor_levels):
                for k in range(num_vec_nnz_levels):
                    x = j+1
                    y = k+1

                    wall_clock_times_success = [wall_clock_time_ for wall_clock_time_ in wall_clock_times[category_id][j][k] if wall_clock_time_ != -1]
                    if len(wall_clock_times_success) > 0:
                        wall_clock_time_result = min(wall_clock_times_success)

                        if best_wall_clock_time_result == -1 or wall_clock_time_result < best_wall_clock_time_result:
                            best_wall_clock_time_result = wall_clock_time_result
                            best_x = x
                            best_y = y

                        if wall_clock_time_result > z_max:
                            print ("wall_clock_time_higher_than_z_max: ", wall_clock_time_result)
                            ax.plot(x, y, 'o', markersize=5, fillstyle='none', color="black")
                        else:
                            x_.append(x)
                            y_.append(y)
                            z_.append(wall_clock_time_result)
                    else:
                        ax.plot(x, y, 'x', markersize=5, color="black")

            norm = colors.LogNorm(vmin=z_min, vmax=z_max)
            SC = ax.scatter(x_, y_, c=z_, norm=norm, cmap=LinearSegmentedColormap.from_list('mymap', [(1,0,0), (0,1,0), (0,0,1), (0,0,0)], N=1024)) #, (0,0,1), (1,0,0), (0,1,0)], N=1024))

            vec_nnz_orig_map = { 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 20, 12: 30, 13: 40, 14: 50, 15: 60, 16: 70, 17: 80, 18: 90, 19: 100 }
            #label = str(round(float(best_wall_clock_time_result),1)) + "s (" + str(best_x) + ", " + str(vec_nnz_orig_map[best_y]) + ")"
            label = str(round(float(best_wall_clock_time_result),1)) + "s"
            ax.plot(best_x, best_y, '*', color='black', label="Best", markersize=5)

            if mattype == "T1" and (category_id==3 or category_id==4):
                ax.annotate(label, # this is the text
                        (best_x,best_y), # this is the point to label
                        color="black",
                        textcoords="offset points", # how to position the text
                        fontsize=16,
                        weight="bold",
                        xytext=(15,-10), # distance from text to points (x,y)
                        ha='center') # horizontal alignment can be left, right or center
            else:
                ax.annotate(label, # this is the text
                        (best_x,best_y), # this is the point to label
                        color="black",
                        textcoords="offset points", # how to position the text
                        fontsize=16,
                        weight="bold",
                        xytext=(5,5), # distance from text to points (x,y)
                        ha='center') # horizontal alignment can be left, right or center

            #if mattype == "T1" and (category_id==3 or category_id==4):
            #    ax.annotate(label, # this is the text
            #            (best_x,best_y), # this is the point to label
            #            color="black",
            #            textcoords="offset points", # how to position the text
            #            backgroundcolor="w",
            #            fontsize=8,
            #            weight="bold",
            #            xytext=(15,-5), # distance from text to points (x,y)
            #            ha='center') # horizontal alignment can be left, right or center
            #else:
            #    ax.annotate(label, # this is the text
            #            (best_x,best_y), # this is the point to label
            #            color="black",
            #            textcoords="offset points", # how to position the text
            #            backgroundcolor="w",
            #            fontsize=8,
            #            weight="bold",
            #            xytext=(0,10), # distance from text to points (x,y)
            #            ha='center') # horizontal alignment can be left, right or center

            fig.add_subplot(ax)

            name_map_rls_method = {
                    "blendenpik": "QR-LSQR",
                    "lsrn": "SVD-LSQR",
                    "newtonsketch": "SVD-PGD"
                    }
            name_map_sketch_operator = {
                    "sjlt": "SJLT",
                    "less_uniform": "LessUnif."
                    }

            if (category_id >= 0 and category_id <= 2):
                ax.set_title(name_map_rls_method[rls_method]) #+ " ("+name_map_sketch_operator[sketch_operator]+")")
            ax.set_ylim([0,20])
            ax.set_xlim([0,11])
            ax.yaxis.set_tick_params(labelsize=8)
            if (category_id >= 3):
                if category_id == 4:
                    ax.set_xlabel('Sampling factor', fontsize=12)
                ax.set_xticks([1,2,3,4,5,6,7,8,9,10])
                ax.set_xticklabels([1,2,3,4,5,6,7,8,9,10])
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])

            #if (category_id == 0 or category_id == 3):
            if (matrix_id == 0 and (category_id == 0 or category_id == 3)):
                ax.set_yticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
                ax.set_yticklabels(["1","2","3","4","5","6","7","8","9","10","20","30","40","50","60","70","80","90","100"])
                if sketch_operator == "less_uniform":
                    ax.set_ylabel("LessUnif. (nnzs per row)", fontsize=12)
                else:
                    ax.set_ylabel("SJLT (nnzs per column)", fontsize=12)
            else:
                ax.set_yticks([])
                ax.set_yticklabels([])
                #ax.set_xticks([])
                #ax.set_xticklabels([])

        fig_title = "Performance of the SAP algorithms (m: "+str(n_rows)+", n: "+str(n_cols)+")"
        fig.suptitle(fig_title, fontsize=15)

        fig.text(0.12, 0.92,
                "Matrix: GA",
                fontsize = 15)

        fig.text(0.36, 0.92,
                "Matrix: T5",
                fontsize = 15)

        fig.text(0.60, 0.92,
                "Matrix: T3",
                fontsize = 15)

        fig.text(0.83, 0.92,
                "Matrix: T1",
                fontsize = 15)

        fig.subplots_adjust(top=0.88, bottom=0.22, left=0.045, right=0.98, wspace=0.05, hspace=0.06)
        cbar = fig.colorbar(SC, cax=fig.add_axes([0.2,0.11,0.65,0.02]), orientation='horizontal') #, anchor=(1.0,0.0))
        cbar.set_label("Wall clock time (s) \n ("+r"$\times$"+" represents failure ("+r"$ARFE > 10 \times ARFE_{ref}$"+ "); $\star$ indicates the best configuration for each category)", fontsize=12)
        cbar.set_ticks([0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,30,40])
        cbar.set_ticklabels([0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,30,40])
        fig.savefig("plots/"+experiment_name+".pdf")

if __name__ == "__main__":

    if not os.path.exists("plots"):
        os.system("mkdir -p plots")

    gen_plot()

