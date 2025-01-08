#!/bin/bash

# Simulate SLURM_ARRAY_TASK_ID loop
for task_id in {1..600}; do
  # Define parameters
  models=("XGB" "MLP" "SVM" "RF" "CNN" "RNN")
  digits=(0 1 2 3 4 5 6 7 8 9)
  replicates_per_job=10
  replicate_start=${1:-1}

  # Precompute values
  num_models=${#models[@]}
  num_digits=${#digits[@]}

  model_index=$(( (task_id - 1) / (num_digits * replicates_per_job) % num_models ))
  digit_index=$(( (task_id - 1) / replicates_per_job % num_digits ))
  replicate_offset=$(( (task_id - 1) % replicates_per_job ))
  replicate=$((replicate_start + replicate_offset))

  # Extract specific parameters
  model=${models[$model_index]}
  digit=${digits[$digit_index]}

  # Debugging information
  echo "Task ID: $task_id"
  echo "Model Index: $model_index, Digit Index: $digit_index, Replicate Offset: $replicate_offset, Replicate: $replicate"
  echo "Mapped to Model: $model, Digit: $digit, Replicate: $replicate"
  echo "----------------------------------------"
done
