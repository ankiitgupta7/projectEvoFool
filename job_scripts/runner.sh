#!/bin/bash --login
#SBATCH --output=logs/exp_%x_%A_%a.out
#SBATCH --error=logs/exp_%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=1-600
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=guptaa23@msu.edu
#SBATCH --job-name=exp_%x_%A_%a  # Base job name

# Guide to Running the Script
# This script runs evolutionary experiments with configurable parameters.
# Usage:
# sbatch runner.sh <experiment> <dataset> <interval> <generations> [replicate_start]
#
# Arguments:
# 1. <experiment>: Experiment number (1, 2_1a, 3)
# 2. <dataset>: Dataset name (e.g., mnistDigits, sklearnDigits, mnistFashion)
# 3. <interval>: Interval for saving generation images (100 for sklearnDigits, 1000 for others)
# 4. <generations>: Number of generations for evolution (50000 for mnistDigits, 100000 for others)
# 5. [replicate_start]: Starting replicate index (optional, default is 1, which will cover 10 (1 to 10) replicates)
#
# Example:
# sbatch runner.sh 1 mnistDigits 1000 100000 11
# This runs experiment 1 on the mnistDigits dataset, with an interval of 1000,
# 100000 generations, and replicates starting from 11 to 20.

# As we are doing 30 replicates for each dataset, model and class, 
# we need host this script 3 times with different replicate_start values, i.e., 1, 11, 21
# also, consider lowering compute time to 20 hours, and memory to 2G

# Ensure required directories exist
mkdir -p logs

# Initialize Conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate pyEnv3.10
export PATH=~/miniforge3/envs/pyEnv3.10/bin:$PATH

# Fixed Parameters
models=("XGB" "MLP" "SVM" "RF" "CNN" "RNN")
classes=(0 1 2 3 4 5 6 7 8 9)
metric="SSIM"

# Input Parameters
experiment=${1:?"Experiment number is required"}
dataset=${2:?"Dataset name is required"}
interval=${3:?"Interval is required"}
generations=${4:?"Number of generations is required"}
replicate_start=${5:-1}

# Derived Parameters
replicates_per_job=10
replicate_end=$((replicate_start + replicates_per_job - 1))
task_id=$SLURM_ARRAY_TASK_ID

# Calculate Indices
num_models=${#models[@]}
num_classes=${#classes[@]}
total_jobs=$((num_models * num_classes * replicates_per_job))

if (( task_id > total_jobs )); then
    echo "Task ID $task_id exceeds total jobs $total_jobs. Exiting."
    exit 1
fi

model_index=$(( (task_id - 1) / (num_classes * replicates_per_job) % num_models ))
class_index=$(( (task_id - 1) / replicates_per_job % num_classes ))
replicate_offset=$(( (task_id - 1) % replicates_per_job ))
replicate=$((replicate_start + replicate_offset))

model=${models[$model_index]}
class=${classes[$class_index]}

# Define a unique job name
job_name="e${experiment}_${dataset}_m${model}_c${class}_r${replicate}_t${task_id}"

# Update SLURM's job name dynamically
scontrol update jobid=$SLURM_JOB_ID JobName="$job_name"

# Output Directory
output_dir="/mnt/home/guptaa23/Active/EPIC_fool/projectEvoFool/output/exp${experiment}_${dataset}${model}_${class}_rep${replicate}_job${SLURM_JOB_ID}"
mkdir -p "$output_dir"

# Debugging Info
echo "Task ID: $task_id, Job ID: $SLURM_JOB_ID"
echo "Experiment: $experiment, Dataset: $dataset, Model: $model, Class: $class, Replicate: $replicate"
echo "Interval: $interval, Generations: $generations"
echo "Output Directory: $output_dir"
env | grep SLURM

# Move to Project Directory
cd /mnt/home/guptaa23/Active/EPIC_fool/projectEvoFool/source

# Run Python Script
exec > "$output_dir/script_output.log" 2> "$output_dir/script_error.log"
python run.py "$experiment" "$dataset" "$model" "$class" "$class" "$metric" "$interval" "$replicate" "$generations"