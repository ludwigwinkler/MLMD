#!/usr/bin/env bash

rm -r IO

#for model in ode rnn lstm bi_ode bi_rnn bi_lstm
#for model in ode rnn lstm hamiltonian bi_ode bi_rnn bi_lstm bi_hamiltonian
for model in lstm bi_lstm;  do
          qsub qsub_modelparallel_MLMD_Interpolation.sh $model
done