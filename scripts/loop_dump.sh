#!/bin/bash


# For max CPU utilitization, each dump file should occupy exactly one CPU. No
# point giving it more. On gadi we need to run this three times due to the max
# queue size of 300.
# Processes reaching the time limit will terminate with error code 271
for i in {0..167}; do
    qsub -v SPLIT=$i -o dump-$i /g/data/v89/at3219/projects/nos/scripts/start_job.sh
    sleep 2
done
