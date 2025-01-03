#!/bin/bash
#SBATCH --output=logs/exp1_%A_%a.out  # Output logs
#SBATCH --error=logs/exp1_%A_%a.err   # Error logs
#SBATCH --time=20:00:00               # Increased time for longer runs
#SBATCH --nodes=1                     # Single node
#SBATCH --ntasks=1                    # One task per job
#SBATCH --cpus-per-task=1             # Number of CPUs per task
#SBATCH --mem=8G                      # Memory per task
#SBATCH --array=1-600                 # Total combinations: 7 models × 10 digits × 10 replicates
#SBATCH --mail-type=END,FAIL           # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=guptaa23@msu.edu  # Your email address


# Initialize Conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate pyEnv3.10  # Activate the Conda environment

# Ensure correct Python is used
export PATH=~/miniforge3/envs/pyEnv3.10/bin:$PATH

# Define fixed parameters
experiment="1"
dataset="skearnDigits"
models=("XGB" "MLP" "SVM" "RF" "CNN" "RNN")
digits=(0 1 2 3 4 5 6 7 8 9)
metric="SSIM"
interval=100
generations=50000

# Configurable replicate range
replicates_per_job=10          # Number of replicates per submission
replicate_start=${1:-1}        # Start of replicate range (passed as argument, default is 1)
replicate_end=$((replicate_start + replicates_per_job - 1))  # End of replicate range

# Calculate indices for task assignment
num_models=${#models[@]}
num_digits=${#digits[@]}
total_jobs=$((num_models * num_digits * replicates_per_job))

# Get task ID from SLURM
task_id=$SLURM_ARRAY_TASK_ID

# Map task ID to parameter combination
model_index=$(( (task_id - 1) / (num_digits * replicates_per_job) % num_models ))
digit_index=$(( (task_id - 1) / replicates_per_job % num_digits ))
replicate_offset=$(( (task_id - 1) % replicates_per_job ))
replicate=$((replicate_start + replicate_offset))

# Extract specific parameters
model=${models[$model_index]}
digit=${digits[$digit_index]}

# Create a dedicated folder for logs and outputs
output_dir="/mnt/home/guptaa23/Active/EPIC_fool/projectEvoFool/output/exp1_${model}_${digit}_rep${replicate}"
mkdir -p "$output_dir"

# Set job name dynamically
scontrol update jobid=$SLURM_JOB_ID JobName="e1_${dataset}_${model}_d${digit}_r${replicate}_${metric}_g${generations}"

# Debugging information
echo "Task ID: $task_id"
echo "Experiment: $experiment, Dataset: $dataset, Model: $model, Digit: $digit, Replicate: $replicate"
echo "Output Directory: $output_dir"

# Move to the project directory
cd /mnt/home/guptaa23/Active/EPIC_fool/projectEvoFool/source

# Run the Python script with the selected parameters
python run.py "$experiment" "$dataset" "$model" "$digit" "$digit" "$metric" "$interval" "$replicate" "$generations" > "$output_dir/output.log" 2> "$output_dir/error.log"