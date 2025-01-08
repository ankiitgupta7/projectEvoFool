#!/bin/bash --login
#SBATCH --output=logs/exp_%x_%A_%a.out
#SBATCH --error=logs/exp_%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=1-10
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=guptaa23@msu.edu
#SBATCH --job-name=exp_%x_%A_%a  # Base job name

# Guide to Running the Script
# This script runs evolutionary experiments for Experiment 3, looping through all 30 replicates for each class in a single job.
# Usage:
# sbatch runner.sh <experiment> <dataset> <interval> <generations>
#
# Arguments:
# 1. <experiment>: Experiment number (3)
# 2. <dataset>: Dataset name (e.g., mnistDigits, sklearnDigits, mnistFashion)
# 3. <interval>: Interval for saving generation images (100 for sklearnDigits, 1000 for others)
# 4. <generations>: Number of generations for evolution (50000 for mnistDigits, 100000 for others)

# Ensure required directories exist
mkdir -p logs

# Initialize Conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate pyEnv3.10
export PATH=~/miniforge3/envs/pyEnv3.10/bin:$PATH

# Fixed Parameters
classes=(0 1 2 3 4 5 6 7 8 9)
metric="SSIM"

# Input Parameters
experiment=${1:?"Experiment number is required"}
dataset=${2:?"Dataset name is required"}
interval=${3:?"Interval is required"}
generations=${4:?"Number of generations is required"}

# Derived Parameters
task_id=$SLURM_ARRAY_TASK_ID

if (( task_id > ${#classes[@]} )); then
    echo "Task ID $task_id exceeds number of classes. Exiting."
    exit 1
fi

class=${classes[$((task_id - 1))]}

# Define a unique job name
job_name="e${experiment}_${dataset}_c${class}_t${task_id}"

# Update SLURM's job name dynamically
scontrol update jobid=$SLURM_JOB_ID JobName="$job_name"

# Debugging Info
echo "Task ID: $task_id, Job ID: $SLURM_JOB_ID"
echo "Experiment: $experiment, Dataset: $dataset, Class: $class"
echo "Interval: $interval, Generations: $generations"
env | grep SLURM

# Move to Project Directory
cd /mnt/home/guptaa23/Active/EPIC_fool/projectEvoFool/source

# Loop through all 30 replicates
for replicate in {1..30}; do
    output_dir="/mnt/home/guptaa23/Active/EPIC_fool/projectEvoFool/output/exp${experiment}_${dataset}_class${class}_rep${replicate}_job${SLURM_JOB_ID}"
    mkdir -p "$output_dir"

    echo "Running Class: $class, Replicate: $replicate"
    echo "Output Directory: $output_dir"

    # Run Python Script for each replicate
    exec > "$output_dir/script_output.log" 2> "$output_dir/script_error.log"
    python run3.py "$experiment" "$dataset" "$class" "$class" "$metric" "$interval" "$replicate" "$generations"
done
