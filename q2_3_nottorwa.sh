#!/bin/bash -l
#PBS -m bea
#PBS -M elb.20@dartmouth.edu
#PBS -N subj24_all
#PBS -l nodes=1:ppn=2
#PBS -l walltime=24:00:00
#PBS -o /dartfs-hpc/scratch/psyc164/mvpaces/log/${PBS_JOBID}.o
#PBS -e /dartfs-hpc/scratch/psyc164/mvpaces/log/${PBS_JOBID}.e

module load python/anaconda2
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate comp_meth_env


cd /dartfs-hpc/scratch/psyc164/mvpaces/code/
./sl_svm_clf.py 12 lh taxonomy
./sl_svm_clf.py 12 lh behavior

