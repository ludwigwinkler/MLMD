#!/usr/bin/env bash

rm -r IO

for model in lstm ode hnn rnn bi_lstm bi_ode bi_hnn bi_rnn; do
#for model in rnn bi_lstm bi_ode bi_hnn bi_rnn; do
for pct in 1.0;do
for dataset in hmc;do
echo ""
echo "##########################################################################################################"
echo ""
python MLMD_Interpolation.py -model $model -dataset $dataset -fast_dev_run True
done
done
done