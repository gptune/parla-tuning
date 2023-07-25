#!/bin/bash

mkdir gptune.log
mkdir lhsmdu.log
mkdir random_search.log
mkdir random_search.db
mkdir tpe.log
mkdir tpe.db

nthreads=8
export OMP_NUM_THREADS=${nthreads}
export MKL_NUM_THREADS=${nthreads}

# a generalized run script

nrun=50
npilot=10
n_rows=50000
n_cols=1000

for batch_num in 1 2 3 4 5
do
    for mattype in "T1" "T3" "T5" "GA"
    do
        python gptune_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -nthreads ${nthreads} -npilot ${npilot} -nrun ${nrun} -batch_num ${batch_num} -failure_handling "highval" | tee gptune.log/log-gptune_search-failure_handling_highval-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-nthreads_${nthreads}-npilot_${npilot}-nrun_${nrun}-batch_num_${batch_num}.txt
        export HYPEROPT_FMIN_SEED=${batch_num}
        python tpe_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -nthreads ${nthreads} -npilot ${npilot} -nrun ${nrun} -batch_num ${batch_num} -failure_handling "highval" | tee tpe.log/log-tpe_search-failure_handling_highval-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-nthreads_${nthreads}-npilot_${npilot}-nrun_${nrun}-batch_num_${batch_num}.txt
        python lhsmdu_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -nthreads ${nthreads} -nrun ${nrun} -batch_num ${batch_num} | tee lhsmdu.log/log-lhsmdu_search-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-nthreads_${nthreads}-nrun_${nrun}-batch_num_${batch_num}.txt
    done
done

## a run script for a distributed computer system
#
#exp=${HOSTNAME}
#
#if [[ ${exp} == "f1" ]]; then
#    nrun=50
#    npilot=10
#    n_rows=50000
#    n_cols=1000
#
#    mattype="T1"
#
#    for batch_num in 1 2 3 4 5
#    do
#        python gptune_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -nthreads ${nthreads} -npilot ${npilot} -nrun ${nrun} -batch_num ${batch_num} -failure_handling "highval" | tee gptune.log/log-gptune_search-failure_handling_highval-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-nthreads_${nthreads}-npilot_${npilot}-nrun_${nrun}-batch_num_${batch_num}.txt
#        export HYPEROPT_FMIN_SEED=${batch_num}
#        python tpe_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -nthreads ${nthreads} -npilot ${npilot} -nrun ${nrun} -batch_num ${batch_num} -failure_handling "highval" | tee tpe.log/log-tpe_search-failure_handling_highval-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-nthreads_${nthreads}-npilot_${npilot}-nrun_${nrun}-batch_num_${batch_num}.txt
#        python lhsmdu_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -nthreads ${nthreads} -nrun ${nrun} -batch_num ${batch_num} | tee lhsmdu.log/log-lhsmdu_search-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-nthreads_${nthreads}-nrun_${nrun}-batch_num_${batch_num}.txt
#    done
#
#elif [[ ${exp} == "f2" ]]; then
#    nrun=50
#    npilot=10
#    n_rows=50000
#    n_cols=1000
#
#    mattype="T3"
#
#    for batch_num in 1 2 3 4 5
#    do
#        python gptune_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -nthreads ${nthreads} -npilot ${npilot} -nrun ${nrun} -batch_num ${batch_num} -failure_handling "highval" | tee gptune.log/log-gptune_search-failure_handling_highval-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-nthreads_${nthreads}-npilot_${npilot}-nrun_${nrun}-batch_num_${batch_num}.txt
#        export HYPEROPT_FMIN_SEED=${batch_num}
#        python tpe_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -nthreads ${nthreads} -npilot ${npilot} -nrun ${nrun} -batch_num ${batch_num} -failure_handling "highval" | tee tpe.log/log-tpe_search-failure_handling_highval-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-nthreads_${nthreads}-npilot_${npilot}-nrun_${nrun}-batch_num_${batch_num}.txt
#        python lhsmdu_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -nthreads ${nthreads} -nrun ${nrun} -batch_num ${batch_num} | tee lhsmdu.log/log-lhsmdu_search-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-nthreads_${nthreads}-nrun_${nrun}-batch_num_${batch_num}.txt
#    done
#
#elif [[ ${exp} == "f3" ]]; then
#    nrun=50
#    npilot=10
#    n_rows=50000
#    n_cols=1000
#
#    mattype="T5"
#
#    for batch_num in 1 2 3 4 5
#    do
#        python gptune_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -nthreads ${nthreads} -npilot ${npilot} -nrun ${nrun} -batch_num ${batch_num} -failure_handling "highval" | tee gptune.log/log-gptune_search-failure_handling_highval-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-nthreads_${nthreads}-npilot_${npilot}-nrun_${nrun}-batch_num_${batch_num}.txt
#        export HYPEROPT_FMIN_SEED=${batch_num}
#        python tpe_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -nthreads ${nthreads} -npilot ${npilot} -nrun ${nrun} -batch_num ${batch_num} -failure_handling "highval" | tee tpe.log/log-tpe_search-failure_handling_highval-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-nthreads_${nthreads}-npilot_${npilot}-nrun_${nrun}-batch_num_${batch_num}.txt
#        python lhsmdu_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -nthreads ${nthreads} -nrun ${nrun} -batch_num ${batch_num} | tee lhsmdu.log/log-lhsmdu_search-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-nthreads_${nthreads}-nrun_${nrun}-batch_num_${batch_num}.txt
#    done
#
#elif [[ ${exp} == "f4" ]]; then
#    nrun=50
#    npilot=10
#    n_rows=50000
#    n_cols=1000
#
#    mattype="GA"
#
#    for batch_num in 1 2 3 4 5
#    do
#        python gptune_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -nthreads ${nthreads} -npilot ${npilot} -nrun ${nrun} -batch_num ${batch_num} -failure_handling "highval" | tee gptune.log/log-gptune_search-failure_handling_highval-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-nthreads_${nthreads}-npilot_${npilot}-nrun_${nrun}-batch_num_${batch_num}.txt
#        export HYPEROPT_FMIN_SEED=${batch_num}
#        python tpe_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -nthreads ${nthreads} -npilot ${npilot} -nrun ${nrun} -batch_num ${batch_num} -failure_handling "highval" | tee tpe.log/log-tpe_search-failure_handling_highval-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-nthreads_${nthreads}-npilot_${npilot}-nrun_${nrun}-batch_num_${batch_num}.txt
#        python lhsmdu_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -nthreads ${nthreads} -nrun ${nrun} -batch_num ${batch_num} | tee lhsmdu.log/log-lhsmdu_search-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-nthreads_${nthreads}-nrun_${nrun}-batch_num_${batch_num}.txt
#    done
#
#else
#    echo "unknown exp ${exp}"
#fi
