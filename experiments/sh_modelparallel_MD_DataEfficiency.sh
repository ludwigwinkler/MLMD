#!/usr/bin/env bash

rm -r IO

#for model in ode rnn lstm bi_ode bi_rnn bi_lstm
for model in ode rnn lstm hamiltonian bi_ode bi_rnn bi_lstm bi_hamiltonian
  do
          qsub qsub_modelparallel_MD_DataEfficiency.sh $model
done
