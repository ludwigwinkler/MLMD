#!/usr/bin/env bash
#$ -l cuda=1   # request one GPU (remove this line if none is needed)
#$ -q all.q    # don't fill the qlogin queue
#$ -cwd        # change working directory (to current)
#$ -V
#$ -e IO/error.txt
#$ -o IO/output.txt

for model in lstm bi_lstm; do
  for output_length_train in 20 10; do
    python MLMD_Interpolation.py -pct_data_set 1.0 -model $model -criterion T -logger False -plot True \
    -input_length 1 -output_length_train $output_length_train -output_length_val 10 -output_length_sampling False\
    -num_hidden 500 -num_layers 5
  done
done
