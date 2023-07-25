#!/bin/bash

mkdir grid_search.log
mkdir grid_search.db

nthreads=8
export OMP_NUM_THREADS=${nthreads}
export MKL_NUM_THREADS=${nthreads}

for mattype in "T1" "T3" "T5" "GA"
do
    for tolerance in 1e-6 1e-8 1e-10
    do
        python grid_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -tolerance ${tolerance} -nthreads ${nthreads} | tee grid_search.log/log-grid_search-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-tolerance_${tolerance}.txt
    done
done

#exp=${HOSTNAME}
#
#if [[ ${exp} == "f1" ]]; then
#    n_rows=50000
#    n_cols=1000
#    mattype="T1"
#    tolerance=1e-6
#    python grid_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -tolerance ${tolerance} -nthreads ${nthreads} | tee grid_search.log/log-grid_search-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-tolerance_${tolerance}.txt
#elif [[ ${exp} == "f2" ]]; then
#    n_rows=50000
#    n_cols=1000
#    mattype="T3"
#    tolerance=1e-6
#    python grid_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -tolerance ${tolerance} -nthreads ${nthreads} | tee grid_search.log/log-grid_search-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-tolerance_${tolerance}.txt
#elif [[ ${exp} == "f3" ]]; then
#    n_rows=50000
#    n_cols=1000
#    mattype="T5"
#    tolerance=1e-6
#    python grid_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -tolerance ${tolerance} -nthreads ${nthreads} | tee grid_search.log/log-grid_search-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-tolerance_${tolerance}.txt
#elif [[ ${exp} == "f4" ]]; then
#    n_rows=50000
#    n_cols=1000
#    mattype="GA"
#    tolerance=1e-6
#    python grid_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -tolerance ${tolerance} -nthreads ${nthreads} | tee grid_search.log/log-grid_search-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-tolerance_${tolerance}.txt
#elif [[ ${exp} == "f5" ]]; then
#    n_rows=50000
#    n_cols=1000
#    mattype="T1"
#    tolerance=1e-8
#    python grid_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -tolerance ${tolerance} -nthreads ${nthreads} | tee grid_search.log/log-grid_search-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-tolerance_${tolerance}.txt
#elif [[ ${exp} == "f6" ]]; then
#    n_rows=50000
#    n_cols=1000
#    mattype="T3"
#    tolerance=1e-8
#    python grid_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -tolerance ${tolerance} -nthreads ${nthreads} | tee grid_search.log/log-grid_search-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-tolerance_${tolerance}.txt
#elif [[ ${exp} == "f7" ]]; then
#    n_rows=50000
#    n_cols=1000
#    mattype="T5"
#    tolerance=1e-8
#    python grid_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -tolerance ${tolerance} -nthreads ${nthreads} | tee grid_search.log/log-grid_search-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-tolerance_${tolerance}.txt
#elif [[ ${exp} == "f8" ]]; then
#    n_rows=50000
#    n_cols=1000
#    mattype="GA"
#    tolerance=1e-8
#    python grid_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -tolerance ${tolerance} -nthreads ${nthreads} | tee grid_search.log/log-grid_search-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-tolerance_${tolerance}.txt
#elif [[ ${exp} == "f9" ]]; then
#    n_rows=50000
#    n_cols=1000
#    mattype="T1"
#    tolerance=1e-10
#    python grid_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -tolerance ${tolerance} -nthreads ${nthreads} | tee grid_search.log/log-grid_search-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-tolerance_${tolerance}.txt
#elif [[ ${exp} == "f10" ]]; then
#    n_rows=50000
#    n_cols=1000
#    mattype="T3"
#    tolerance=1e-10
#    python grid_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -tolerance ${tolerance} -nthreads ${nthreads} | tee grid_search.log/log-grid_search-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-tolerance_${tolerance}.txt
#elif [[ ${exp} == "f11" ]]; then
#    n_rows=50000
#    n_cols=1000
#    mattype="T5"
#    tolerance=1e-10
#    python grid_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -tolerance ${tolerance} -nthreads ${nthreads} | tee grid_search.log/log-grid_search-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-tolerance_${tolerance}.txt
#elif [[ ${exp} == "f12" ]]; then
#    n_rows=50000
#    n_cols=1000
#    mattype="GA"
#    tolerance=1e-10
#    python grid_search.py -n_rows ${n_rows} -n_cols ${n_cols} -mattype ${mattype} -tolerance ${tolerance} -nthreads ${nthreads} | tee grid_search.log/log-grid_search-n_rows_${n_rows}-n_cols_${n_cols}-mattype_${mattype}-tolerance_${tolerance}.txt
#else
#    echo "unknown exp ${exp}"
#fi

