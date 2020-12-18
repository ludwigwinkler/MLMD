#!/usr/bin/env bash

rm -r IO

#for model in ode rnn lstm bi_ode bi_rnn bi_lstm
#for model in hamiltonian ode rnn lstm bi_ode bi_rnn bi_lstm bi_hamiltonian
for model in bi_lstm; do
for pct in 1.0;do
for data in benzene_dft.npz;do
qsub qsub_MLMD_Interpolation.sh $pct $model $data
done
done
done