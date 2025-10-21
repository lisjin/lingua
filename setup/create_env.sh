#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

#SBATCH --job-name=env_creation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --qos=lowest
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Exit immediately if a command exits with a non-zero status
set -e

# Start timer
start_time=$(date +%s)

# Get the current date
current_date=$(date +%y%m%d)

# Create environment name with the current date
env_prefix=lingua_$current_date

# Create the conda environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda create -n $env_prefix python=3.11 -y
conda activate $env_prefix

echo "Currently in env $(which python)"

# Install packages
pip install torch==2.7.0 xformers
pip install ninja
pip install --requirement requirements.txt

# End timer
end_time=$(date +%s)

# Calculate elapsed time in seconds
elapsed_time=$((end_time - start_time))

# Convert elapsed time to minutes
elapsed_minutes=$((elapsed_time / 60))

echo "Environment $env_prefix created and all packages installed successfully in $elapsed_minutes minutes!"

