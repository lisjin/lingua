#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

#SBATCH --job-name=env_creation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --account=fair_amaia_cw_explore
#SBATCH --qos=lowest
#SBATCH --time=3:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Exit immediately if a command exits with a non-zero status
set -e

# Start timer
start_time=$(date +%s)

env_prefix=${CKPT_HOME}/envs/lingua-parq
if [ ! -d "$env_prefix" ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
    conda create -p $env_prefix python=3.11 -y
fi
conda activate $env_prefix

pip install --upgrade pip setuptools
pip install ninja
pip install 'torch==2.7.0' 'xformers==0.0.30' --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip uninstall pynvml -y

if [ ! -d $HOME/qpat ]; then
    cd $HOME && git clone git@github.com:fairinternal/qpat.git
fi
cd $HOME/qpat && pip install --no-dependencies -e '.[dev]'

# End timer
end_time=$(date +%s)

# Calculate elapsed time in seconds
elapsed_time=$((end_time - start_time))

# Convert elapsed time to minutes
elapsed_minutes=$((elapsed_time / 60))

echo "Environment $env_prefix created and all packages installed successfully in $elapsed_minutes minutes!"

