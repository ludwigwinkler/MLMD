#!/usr/bin/env bash
#$ -l cuda=1   # request one GPU (remove this line if none is needed)
#$ -q all.q    # don't fill the qlogin queue
#$ -cwd        # change working directory (to current)
#$ -V
#$ -e IO/error.txt
#$ -o IO/output.txt

for model in lstm bi_lstm; do
  for output_length_train in 10 20; do
    python MLMD_Interpolation.py -pct_data_set $pct -model $1 -criterion $criterion -logger True -data_set $data -plot False \
    -input_length 1 -output_length_train $output_length -output_length_val 10 \
    -num_hidden 500 -num_layers 5
  done
done
