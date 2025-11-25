#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=8:00:00
#SBATCH --account=parq
#SBATCH --qos=dev

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate lingua

data_dir=/fsx-parq/shared/datasets
python $HOME/lingua/setup/download_prepare_hf_data.py dclm_baseline_1.0_10prct 8 --data_dir $data_dir --seed 42 --nchunks 8
