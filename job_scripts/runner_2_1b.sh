#!/bin/bash --login
#SBATCH --output=logs/exp_%x_%A_%a.out
#SBATCH --error=logs/exp_%x_%A_%a.err
#SBATCH --time=15:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1  # Request 1 GPU
#SBATCH --mem=8G
#SBATCH --array=1-200
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=guptaa23@msu.edu
#SBATCH --job-name=exp_%x_%A_%a  # Base job name


# Guide to Running the Script
# This script runs evolutionary experiments with configurable parameters.
# Usage:
# sbatch runner.sh <experiment> <dataset> <interval> <generations> [replicate_start]
#
# Arguments:
# 1. <experiment>: Experiment number (2_1b)
# 2. <dataset>: Dataset name (e.g., mnistDigits, sklearnDigits, mnistFashion)
# 3. <interval>: Interval for saving generation images (100 for sklearnDigits, 1000 for others)
# 4. <generations>: Number of generations for evolution (50000 for mnistDigits, 100000 for others)
# 5. [replicate_start]: Starting replicate index (optional, default is 1, which will cover 10 (1 to 10) replicates)
#
# Example:
# sbatch runner_2_1b.sh 2_1b mnistDigits 1000 100000 11
# sbatch runner_2_1b.sh 2_1b sklearnDigits 100 50000 1
# This runs experiment 2_1b on the mnistDigits dataset, with an interval of 1000,
# 100000 generations, and replicates starting from 11 to 20.

# As we are doing 30 replicates for each dataset, model, and class, 
# we need to host this script 3 times with different replicate_start values, i.e., 1, 11, 21
# also, consider lowering compute time to 20 hours, and memory to 2G in case of sklearnDigits

# Ensure required directories exist
mkdir -p logs

# Initialize Conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate pyEnv3.10
export PATH=~/miniforge3/envs/pyEnv3.10/bin:$PATH

# Fixed Parameters
models=("CNN" "RNN")
metric="SSIM"
random_seed=42  # Seed for reproducibility

# Randomly generated class pairs for seed 42
targets=(9 1 0 3 3 3 1 1 9 7)
non_targets=(0 6 4 9 5 1 9 5 5 6)

# Input Parameters
experiment=${1:?-"Experiment number is required"}
dataset=${2:?-"Dataset name is required"}
interval=${3:?-"Interval is required"}
generations=${4:?-"Number of generations is required"}
replicate_start=${5:-1}

# Derived Parameters
replicates_per_class=10
num_class_combinations=${#targets[@]}  # Number of class combinations (10)
jobs_per_model=$((num_class_combinations * replicates_per_class))  # Jobs per model (10 * 10 = 100)
num_models=${#models[@]}  # Total number of models (4)
total_jobs=$((jobs_per_model * num_models))  # Total jobs across all models (400)

# Check if task ID exceeds total jobs
if (( SLURM_ARRAY_TASK_ID > total_jobs )); then
    echo "Task ID $SLURM_ARRAY_TASK_ID exceeds total jobs $total_jobs. Exiting."
    exit 1
fi

# Calculate the model index and task offset within the model
model_index=$(( (SLURM_ARRAY_TASK_ID - 1) / jobs_per_model ))  # Determine which model the task ID corresponds to
task_offset=$(( (SLURM_ARRAY_TASK_ID - 1) % jobs_per_model ))  # Offset within the current model's jobs

# Calculate the replicate offset and class combination index
class_combination_index=$(( task_offset / replicates_per_class ))  # Class combination index (0-9)
replicate_offset=$(( task_offset % replicates_per_class ))  # Replicate index within the class combination (0-9)

# Assign model, target, and non-target classes
model=${models[$model_index]}
target_class=${targets[$class_combination_index]}
non_target_class=${non_targets[$class_combination_index]}
replicate=$((replicate_start + replicate_offset))

# Define a unique job name
job_name="e${experiment}_${dataset}_m${model}_tc${target_class}_ntc${non_target_class}_r${replicate}_t${SLURM_ARRAY_TASK_ID}"

# Update SLURM's job name dynamically
scontrol update jobid=$SLURM_JOB_ID JobName="$job_name"

# Output Directory
output_dir="/mnt/home/guptaa23/Active/EPIC_fool/projectEvoFool/output/exp${experiment}_${dataset}${model}_tc${target_class}_ntc${non_target_class}_rep${replicate}_job${SLURM_JOB_ID}"
mkdir -p "$output_dir"

# Debugging Info
echo "Task ID: $SLURM_ARRAY_TASK_ID, Model Index: $model_index, Task Offset: $task_offset"
echo "Class Combination Index: $class_combination_index, Replicate Offset: $replicate_offset"
echo "Model: $model, Target Class: $target_class, Non-Target Class: $non_target_class, Replicate: $replicate"
echo "Output Directory: $output_dir"
env | grep SLURM

# Move to Project Directory
cd /mnt/home/guptaa23/Active/EPIC_fool/projectEvoFool/source

# Run Python Script
exec > "$output_dir/script_output.log" 2> "$output_dir/script_error.log"
python run.py "$experiment" "$dataset" "$model" "$target_class" "$non_target_class" "$metric" "$interval" "$replicate" "$generations"