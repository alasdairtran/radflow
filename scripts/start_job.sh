#!/bin/bash
#PBS -l ncpus=4
#PBS -l walltime=48:00:00
#PBS -l mem=16GB
#PBS -l jobfs=20MB
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
conda activate radflow
cd /g/data/v89/at3219/projects/radflow
DUMP=/g/data/v89/at3219/wikidump
OUT=/g/data/v89/at3219/projects/radflow/results

python scripts/extract_graph.py -s $SPLIT -d $DUMP -o $OUT -n 4
