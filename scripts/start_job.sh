#!/bin/bash
#PBS -l ncpus=2
#PBS -l walltime=48:00:00
#PBS -l mem=8GB
#PBS -l jobfs=10MB
#PBS -P v89
#PBS -q normal
#PBS -l other=gdata1
#PBS -l storage=scratch/v89+gdata/v89
#PBS -M alasdair.tran@anu.edu.au
#PBS -N extract-wiki
#PBS -j oe
#PBS -m abe
#PBS -l wd

source $HOME/.bashrc
conda activate nos
cd /g/data/v89/at3219/projects/nos
DUMP=/g/data/v89/at3219/wikidump2
OUT=/g/data/v89/at3219/projects/nos/results3

python scripts/extract_graph.py -i $ORDER -d $DUMP -o $OUT
