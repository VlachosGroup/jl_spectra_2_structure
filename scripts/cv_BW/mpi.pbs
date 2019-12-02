#!/bin/bash
#PBS -j oe
#PBS -q low
#PBS -l nodes=399:ppn=32:xe
#PBS -l walltime=24:00:00

cd $PBS_O_WORKDIR

module load bwpy/2.0.2
module load bwpy-mpi

NRANKS=$(wc -l <$PBS_NODEFILE)
cpus=`expr $NRANKS / 4`
echo $cpus
aprun -n 396 -N 1 -b python ./mpicommexecutor.py 
