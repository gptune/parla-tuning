#!/bin/bash

mkdir lhsmdu.log
mkdir lhsopenturns.log

nthreads=8
export OMP_NUM_THREADS=${nthreads}
export MKL_NUM_THREADS=${nthreads}

nrun=100
batch_num=1

# a generalized run script
 
n_rows=10000
n_cols=1000

for mattype in "T1" "T3" "T5" "GA"
do
    python lhs_search.py -search_module "LHSMDU" -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -nthreads ${nthreads} -nrun ${nrun} -batch_num ${batch_num} -failure_handling "highval" | tee lhsmdu.log/log-lhsmdu_search-failure_handling_highval-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-nthreads_${nthreads}-nrun_${nrun}-batch_num_${batch_num}.txt
done

## a run script for a distributed computer system
#
#exp=${HOSTNAME}
#
#if [[ ${exp} == "f1" ]]; then
#    n_rows=10000
#    n_cols=1000
#
#    mattype="T1"
#
#    python lhs_search.py -search_module "LHSMDU" -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -nthreads ${nthreads} -nrun ${nrun} -batch_num ${batch_num} -failure_handling "highval" | tee lhsmdu.log/log-lhsmdu_search-failure_handling_highval-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-nthreads_${nthreads}-nrun_${nrun}-batch_num_${batch_num}.txt
#
#elif [[ ${exp} == "f2" ]]; then
#    n_rows=10000
#    n_cols=1000
#
#    mattype="T3"
#
#    python lhs_search.py -search_module "LHSMDU" -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -nthreads ${nthreads} -nrun ${nrun} -batch_num ${batch_num} -failure_handling "highval" | tee lhsmdu.log/log-lhsmdu_search-failure_handling_highval-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-nthreads_${nthreads}-nrun_${nrun}-batch_num_${batch_num}.txt
#
#elif [[ ${exp} == "f3" ]]; then
#    n_rows=10000
#    n_cols=1000
#
#    mattype="T5"
#
#    python lhs_search.py -search_module "LHSMDU" -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -nthreads ${nthreads} -nrun ${nrun} -batch_num ${batch_num} -failure_handling "highval" | tee lhsmdu.log/log-lhsmdu_search-failure_handling_highval-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-nthreads_${nthreads}-nrun_${nrun}-batch_num_${batch_num}.txt
#
#elif [[ ${exp} == "f4" ]]; then
#    n_rows=10000
#    n_cols=1000
#
#    mattype="GA"
#
#    python lhs_search.py -search_module "LHSMDU" -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -nthreads ${nthreads} -nrun ${nrun} -batch_num ${batch_num} -failure_handling "highval" | tee lhsmdu.log/log-lhsmdu_search-failure_handling_highval-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-nthreads_${nthreads}-nrun_${nrun}-batch_num_${batch_num}.txt
#
#else
#    echo "unknown exp ${exp}"
#fi
