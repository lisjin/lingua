#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=lingua-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --account=fair_amaia_cw_explore
#SBATCH --qos=lowest
#SBATCH --array=0
#SBATCH --output=/checkpoint/parq/%u/lingua_dumps/lingua-debug_%A/%a/train.out
#SBATCH --error=/checkpoint/parq/%u/lingua_dumps/lingua-debug_%A/%a/train.err
#SBATCH --time=2:00:00

source ~/.profile

head_node=`scontrol show hostnames $SLURM_JOB_NODELIST | sed 1q`
head_node_ip=`srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address`
echo Node IP: $head_node_ip

model_name=lingua-debug_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
save_dir=/checkpoint/parq/${USER}/lingua_dumps/${model_name}/${SLURM_ARRAY_TASK_ID}
mkdir -p $save_dir

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $CKPT_HOME/envs/lingua-parq

reg_lmbdas=(1e-2)
nnodes=1
nproc_per_node=8
batch_size=$((256 / ($nproc_per_node * $nnodes)))
seed=$RANDOM
HF_DATASETS_OFFLINE=1 srun torchrun \
  --nnodes 1 \
  --nproc-per-node $nproc_per_node \
  --rdzv-id $seed \
  --rdzv-backend c10d \
  --rdzv-endpoint $head_node_ip:29500 \
  -m apps.main.train \
    name=$model_name \
    seed=$seed \
    data.root_dir=/datasets/llm/pretraining \
    data.batch_size=$batch_size \
    dump_dir=$save_dir \
    config=apps/main/configs/debug.yaml \
    logging.wandb.project=lingua \
    logging.wandb.id=$model_name \
    logging.wandb.dir=$CKPT_HOME \
    optim.warmup=250 \
    optim.prune_reg_lambda=${reg_lmbdas[$SLURM_ARRAY_TASK_ID]} \
    optim.prune_config_path=apps/main/configs/prune/lasso_linear.yaml \
    optim.prune_warmup_steps=250 \
