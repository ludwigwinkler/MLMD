#!/usr/bin/env bash

rm -r IO

#for model in ode rnn lstm bi_ode bi_rnn bi_lstm
#for model in ode rnn lstm hamiltonian bi_ode bi_rnn bi_lstm bi_hamiltonian
for model in lstm;  do
#  for data in keto_100K_0.2fs.npz keto_300K_0.2fs.npz keto_500K_0.2fs.npz; do
#  for data in keto_300K_0.2fs.npz ethanol_dft.npz salicylic_dft.npz keto_100K_0.2fs.npz; do
    qsub qsub_modelparallel_MLMD_Interpolation.sh $model
#  done
done
