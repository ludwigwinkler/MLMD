#!/usr/bin/env bash

#for model in ode rnn lstm bi_ode bi_rnn bi_lstm
#for model in hamiltonian ode rnn lstm bi_ode bi_rnn bi_lstm bi_hamiltonian
for model in bi_ode
  do
    for pct in 0.001 0.01 0.1 0.25 0.5 1.0
      do
        for data in keto_0.2fs.npz
          do
          qsub qsub_MD_DataEfficiency.sh $pct $model $data
done
done
done