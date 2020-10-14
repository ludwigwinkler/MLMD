#!/usr/bin/env bash
#$ -l cuda=1   # request one GPU (remove this line if none is needed)
#$ -l h=!(node28|node01)   # Do not place job on node28 which doesn't support neweset cuda
#$ -q all.q    # don't fill the qlogin queue
#$ -cwd        # change working directory (to current)
#$ -V
#$ -e IO/error.txt
#$ -o IO/output.txt

python MD_DataEfficiency.py -pct_train_data $1 -model $2 -logger True -train_data $3 -plot False -output_length 5

