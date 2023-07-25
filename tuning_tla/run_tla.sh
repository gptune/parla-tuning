#!/bin/bash

mkdir gptune_tla.log

nthreads=8
export OMP_NUM_THREADS=${nthreads}
export MKL_NUM_THREADS=${nthreads}

nrun=50

# a generalized run script

n_rows=50000
n_cols=1000

source_n_rows=10000
source_n_cols=1000

failure_handling="highval"
source_failure_handling="highval"

for batch_num in 1 2 3 4 5
do
    for mattype in "T1" "T3" "T5" "GA"
    do
        for source_mattype in "T1" "T3" "T5" "GA"
        do
            python gptune_tla_original.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -failure_handling ${failure_handling} -nthreads ${nthreads} -nrun ${nrun} -batch_num ${batch_num} -source_mattype ${source_mattype} -source_n_rows ${source_n_rows} -source_n_cols ${source_n_cols} -source_failure_handling ${source_failure_handling} | tee gptune_tla.log/log-gptune_tla_original-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-source_n_rows_${source_n_rows}-source_n_cols_${source_n_cols}-source_mattype_${source_mattype}-nthreads_${nthreads}-nrun_${nrun}-batch_num_${batch_num}.txt
        done
    done
done

## a run script for a distributed computer system
#
#exp=${HOSTNAME}
#
#if [[ ${exp} == "f1" ]]; then
#    n_rows=50000
#    n_cols=1000
#
#    source_n_rows=10000
#    source_n_cols=1000
#
#    failure_handling="highval"
#    source_failure_handling="highval"
#
#    mattype="T1"
#    for batch_num in 1 2 3 4 5
#    do
#        for source_mattype in "T1" "T3" "T5" "GA"
#        do
#            python gptune_tla_original.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -failure_handling ${failure_handling} -nthreads ${nthreads} -nrun ${nrun} -batch_num ${batch_num} -source_mattype ${source_mattype} -source_n_rows ${source_n_rows} -source_n_cols ${source_n_cols} -source_failure_handling ${source_failure_handling} | tee gptune_tla.log/log-gptune_tla_original-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-source_n_rows_${source_n_rows}-source_n_cols_${source_n_cols}-source_mattype_${source_mattype}-nthreads_${nthreads}-nrun_${nrun}-batch_num_${batch_num}.txt
#        done
#    done
#
#elif [[ ${exp} == "f2" ]]; then
#    n_rows=50000
#    n_cols=1000
#
#    source_n_rows=10000
#    source_n_cols=1000
#
#    failure_handling="highval"
#    source_failure_handling="highval"
#
#    mattype="T3"
#    for batch_num in 1 2 3 4 5
#    do
#        for source_mattype in "T1" "T3" "T5" "GA"
#        do
#            python gptune_tla_original.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -failure_handling ${failure_handling} -nthreads ${nthreads} -nrun ${nrun} -batch_num ${batch_num} -source_mattype ${source_mattype} -source_n_rows ${source_n_rows} -source_n_cols ${source_n_cols} -source_failure_handling ${source_failure_handling} | tee gptune_tla.log/log-gptune_tla_original-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-source_n_rows_${source_n_rows}-source_n_cols_${source_n_cols}-source_mattype_${source_mattype}-nthreads_${nthreads}-nrun_${nrun}-batch_num_${batch_num}.txt
#        done
#    done
#
#elif [[ ${exp} == "f3" ]]; then
#    n_rows=50000
#    n_cols=1000
#
#    source_n_rows=10000
#    source_n_cols=1000
#
#    failure_handling="highval"
#    source_failure_handling="highval"
#
#    mattype="T5"
#    for batch_num in 1 2 3 4 5
#    do
#        for source_mattype in "T1" "T3" "T5" "GA"
#        do
#            python gptune_tla_original.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -failure_handling ${failure_handling} -nthreads ${nthreads} -nrun ${nrun} -batch_num ${batch_num} -source_mattype ${source_mattype} -source_n_rows ${source_n_rows} -source_n_cols ${source_n_cols} -source_failure_handling ${source_failure_handling} | tee gptune_tla.log/log-gptune_tla_original-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-source_n_rows_${source_n_rows}-source_n_cols_${source_n_cols}-source_mattype_${source_mattype}-nthreads_${nthreads}-nrun_${nrun}-batch_num_${batch_num}.txt
#        done
#    done
#
#elif [[ ${exp} == "f4" ]]; then
#    n_rows=50000
#    n_cols=1000
#
#    source_n_rows=10000
#    source_n_cols=1000
#
#    failure_handling="highval"
#    source_failure_handling="highval"
#
#    mattype="GA"
#    for batch_num in 1 2 3 4 5
#    do
#        for source_mattype in "T1" "T3" "T5" "GA"
#        do
#            python gptune_tla_original.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -failure_handling ${failure_handling} -nthreads ${nthreads} -nrun ${nrun} -batch_num ${batch_num} -source_mattype ${source_mattype} -source_n_rows ${source_n_rows} -source_n_cols ${source_n_cols} -source_failure_handling ${source_failure_handling} | tee gptune_tla.log/log-gptune_tla_original-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-source_n_rows_${source_n_rows}-source_n_cols_${source_n_cols}-source_mattype_${source_mattype}-nthreads_${nthreads}-nrun_${nrun}-batch_num_${batch_num}.txt
#        done
#    done
#
#else
#    echo "unknown exp ${exp}"
#fi
