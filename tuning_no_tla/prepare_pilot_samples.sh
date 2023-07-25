#!/bin/bash

nthreads=8
export OMP_NUM_THREADS=${nthreads}
export MKL_NUM_THREADS=${nthreads}

n_rows=50000
n_cols=1000
n_samples=10

for batch_num in 1 2 3 4 5
do
    python lhsmdu_gen.py -n_rows ${n_rows} -n_cols ${n_cols} -n_samples ${n_samples} -batch_num ${batch_num}
done

