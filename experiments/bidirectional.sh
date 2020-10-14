#!/usr/bin/env bash
#$ -l cuda=1   # request one GPU (remove this line if none is needed)
#$ -q all.q    # don't fill the qlogin queue
#$ -cwd        # change working directory (to current)
#$ -V
#$ -e IO/error.txt
#$ -o IO/output.txt


for model in bi_rnn; do
  for output_length in 10; do
    python MD_DataEfficiency.py
  done
done
