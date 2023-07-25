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

    experiment_name = "analysis_motivation"

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16
    fig, ax = plt.subplots(1, 2, figsize=(12,4))

    z_min = 0.5
    z_max = 40

    for matrix_id in range(2):
        if matrix_id == 0:
            mattype = "GA"
        #elif matrix_id == 1:
        #    mattype = "T5"
        #elif matrix_id == 2:
        #    mattype = "T3"
        elif matrix_id == 1:
            mattype = "T1"

        reference_configuration = {}

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

                    reference_configuration = {
                        "label": "S-5-50",
                        "wall_clock_time": func_eval["evaluation_result"]["wall_clock_time"],
                        "ARFE": reference_normalized_residual_error_to_Axstar
                    }

        configurations = []

        best_x, best_y, best_wall_clock_time_result = -1, -1, -1

        #for tolerance in [1e-6, 1e-8, 1e-10]:
        for tolerance in [1e-6]: #, 1e-8, 1e-10]:
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

                if rls_method == "blendenpik" and\
                   sketch_operator == "less_uniform" and\
                   sampling_factor == 1 and\
                   vec_nnz == 1:
                    configurations.append({
                        "label": "1k x 50k (nnz: 1)",
                        "wall_clock_time": func_eval["evaluation_result"]["wall_clock_time"],
                        "ARFE": normalized_residual_error_to_Axstar
                    })

                if rls_method == "blendenpik" and\
                   sketch_operator == "less_uniform" and\
                   sampling_factor == 5 and\
                   vec_nnz == 1:
                    configurations.append({
                        "label": "5k x 50k (nnz: 1)",
                        "wall_clock_time": func_eval["evaluation_result"]["wall_clock_time"],
                        "ARFE": normalized_residual_error_to_Axstar
                    })

                if rls_method == "blendenpik" and\
                   sketch_operator == "less_uniform" and\
                   sampling_factor == 5 and\
                   vec_nnz == 10:
                    configurations.append({
                        "label": "5k x 50k (nnz: 10)",
                        "wall_clock_time": func_eval["evaluation_result"]["wall_clock_time"],
                        "ARFE": normalized_residual_error_to_Axstar
                    })

                if rls_method == "blendenpik" and\
                   sketch_operator == "less_uniform" and\
                   sampling_factor == 5 and\
                   vec_nnz == 100:
                    configurations.append({
                        "label": "5k x 50k (nnz: 100)",
                        "wall_clock_time": func_eval["evaluation_result"]["wall_clock_time"],
                        "ARFE": normalized_residual_error_to_Axstar
                    })

        x = [ configurations[i]["label"] for i in range(len(configurations))]
        y = [ configurations[i]["wall_clock_time"] for i in range(len(configurations))]
        y2 = [ configurations[i]["ARFE"] for i in range(len(configurations))]

        if mattype == "GA":
            #ax[matrix_id].set_title("Matrix with nearly uniform leverage scores")
            ax[matrix_id].set_title("$\mathsf{GA}$ matrix (50k x 1k) (see Table 3)", fontsize=16)
            #ax[matrix_id].set_title("Matrix (50k x 1k) with \n nearly uniform leverage scores", fontsize=16)
        elif mattype == "T1":
            ax[matrix_id].set_title("$\mathsf{T1}$ matrix (50k x 1k) (see Table 3)", fontsize=16)
            #ax[matrix_id].set_title("Matrix (50k x 1k) with \n very nonuniform leverage scores", fontsize=16)
        #ax[matrix_id].set_title("Input matrix: "+mattype + " (m: 50,000, n: 1,000)")
        ax[matrix_id].bar(x, y, color="white", edgecolor="black", label="Wall-clock time")

        #ax[matrix_id].plot(x, y, color="gray", label="bar")
        ax[matrix_id].set_ylabel("Wall-clock time (s)", fontsize=16)
        ax[matrix_id].set_xticklabels(ax[matrix_id].get_xticklabels(), rotation=20)
        if matrix_id == 0:
            #ax[matrix_id].set_yticks([0,1,2,3,4,5,6])
            ax[matrix_id].set_yticks([0,0.5,1.0,1.5])
        if matrix_id == 1:
            ax[matrix_id].set_ylim([0, 20])
        #ax[matrix_id].set_yticks([1,2,3,4,5,6,7,8])
        #ax[matrix_id].plot(x, y, linestyle="-", marker="o", color="blue")
        ax[matrix_id].set_xlabel("Sketching matrix configuration under LessUniform (see Sec. 3.2)", fontsize=16)
        ax_twin = ax[matrix_id].twinx()
        #ax_twin.plot(x, y2, "-", legend="ASDF") #color="red", legend="ASDF")
        #ax_twin.plot(x, y2, marker="o", linestyle= "-", label="ARFE") #, legend="ASDF") #color="red", legend="ASDF")
        ax_twin.plot(x, y2, marker="o", label="Residual error ($\mathit{ARFE}$) \n(see Sec. 4.1.2)") #, legend="ASDF") #color="red", legend="ASDF")

        if matrix_id == 0:
            ax[matrix_id].legend(frameon=False, bbox_to_anchor=(0.8, 1.05))
            ax_twin.legend(frameon=False, bbox_to_anchor=(0.25, 0.9))

        ax_twin.set_yscale("log")
        ax_twin.set_yticks([1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])
        #ax_twin.set_ylabel("$ARFE$")
        ax_twin.set_ylabel("Residual error", fontsize=16)

    fig.subplots_adjust(top=0.80, bottom=0.35, left=0.06, right=0.94, wspace=0.4, hspace=0.1)
    fig.suptitle("Performance of an SAP algorithm ($\mathsf{QR-LSQR}$) (see Table 1 for details)", fontsize=16)
    fig.savefig("plots/"+experiment_name+".pdf")

if __name__ == "__main__":

    if not os.path.exists("plots"):
        os.system("mkdir -p plots")

    gen_plot()

