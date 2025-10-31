#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

#SBATCH --job-name=env_creation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --account=fair_amaia_cw_explore
#SBATCH --qos=lowest
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Exit immediately if a command exits with a non-zero status
set -e

# Start timer
start_time=$(date +%s)

env_prefix=${CKPT_HOME}/envs/lingua-parq
uv venv $env_prefix --python 3.13
source ${env_prefix}/bin/activate

echo "Currently in env $(which python)"

# Install packages
uv pip install xformers --index-url https://download.pytorch.org/whl/cu126
uv pip install ninja
uv pip install -r requirements.txt

# End timer
end_time=$(date +%s)

# Calculate elapsed time in seconds
elapsed_time=$((end_time - start_time))

# Convert elapsed time to minutes
elapsed_minutes=$((elapsed_time / 60))

echo "Environment $env_prefix created and all packages installed successfully in $elapsed_minutes minutes!"

