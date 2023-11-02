#!/bin/bash

#############################
# Your job name (displayed by the queue)

#SBATCH -J Your_experiment_name

#SBATCH --array=0-89
# walltime (hh:mm::ss)
#SBATCH -t 20:10:00

# Specify the number of nodes(nodes=) and the number of cores per nodes(tasks-pernode=) to be used
#SBATCH - N 1
#SBATCH --ntasks-per-node=1

# change working directory
# SBATCH --chdir=.

#############################

env_id="$1"
h_test="multi_res_cmaes"
nb_trials=10
sampler="Cmaes"
nb_res=3

ENV_ID_JOB="${env_id}_${h_test}"

# useful informations to print
echo "#############################"
echo "User:" $USER
echo "Date:" `date`
echo "Host:" `hostname`
echo "Directory:" `pwd`
echo "SLURM_JOBID:" $SLURM_JOBID
echo "SLURM_SUBMIT_DIR:" $SLURM_SUBMIT_DIR
echo "SLURM_JOB_NODELIST:" $SLURM_JOB_NODELIST
echo "#############################"

#############################

# What you actually want to launch

python3 HP_multi_res.py --nb_trials $nb_trials --env_id $env_id --nb_res $nb_res --h_test $h_test --training_steps 300000 --sampler $sampler

# all done
echo "Job finished"