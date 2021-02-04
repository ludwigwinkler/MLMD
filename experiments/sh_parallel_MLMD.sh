#!/usr/bin/env bash

rm -r IO

for model in bi_lstm; do
  for dataset in ethanol_dft.npz paracetamol_dft.npz aspirin_dft.npz; do
    for output_length in 20; do
      qsub qsub_parallel_MLMD.sh $model $dataset
    done
  done
done
