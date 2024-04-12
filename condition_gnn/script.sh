#!/bin/bash
#SBATCH --job-name=array
#SBATCH --output=experiments/logs/array_er_%A_%a.out
#SBATCH --error=experiments/logs/array_er_%A_%a.err
#SBATCH --array=1-1
#SBATCH --time=1:00:00
#SBATCH --partition=caslake
#SBATCH --mem=10G
#SBATCH --account=pi-cdonnat

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "My SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

# Add lines here to run your computations
job_id=$SLURM_ARRAY_JOB_ID


result_file="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "result file is ${result_file}"
cd $SCRATCH/$USER/GNN/condition

module load gsl
module load gcc
module load python/anaconda-2021.05
python3 main.py  --namefile $result_file --seed $SLURM_ARRAY_TASK_ID --dataset $1 
\

