#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=llama3.2-1b
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=1600G
#SBATCH --account=parq
#SBATCH --qos=h200_parq_high
#SBATCH --array=2-5
#SBATCH --output=/checkpoint/parq/%u/lingua_dumps/llama3.2-1b_%A/%a/train.out
#SBATCH --error=/checkpoint/parq/%u/lingua_dumps/llama3.2-1b_%A/%a/train.err
#SBATCH --time=2-00:00:00
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120
#SBATCH --distribution=block

source ~/.profile

head_node=`scontrol show hostnames $SLURM_JOB_NODELIST | sed 1q`
head_node_ip=`srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address`
node_port=$((29000 + (SLURM_ARRAY_JOB_ID % 1000)))
echo Host node address: ${head_node_ip}:${node_port}

model_name=llama3.2-1b_${SLURM_ARRAY_JOB_ID}
save_dir=/checkpoint/parq/${USER}/lingua_dumps/${model_name}/${SLURM_ARRAY_TASK_ID}
model_name+=_${SLURM_ARRAY_TASK_ID}
mkdir -p $save_dir

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $CKPT_HOME/envs/lingua-parq

reg_lmbdas=(2e-4 5e-4 1e-3 2e-3 5e-3 1e-2)
nproc_per_node=8
tot_batch_size=128
batch_size=$((tot_batch_size / (nproc_per_node * SLURM_JOB_NUM_NODES)))
seed=$RANDOM
HF_DATASETS_OFFLINE=1 srun torchrun \
  --nnodes $SLURM_JOB_NUM_NODES \
  --nproc-per-node $nproc_per_node \
  --rdzv-id $seed \
  --rdzv-backend c10d \
  --rdzv-endpoint ${head_node_ip}:${node_port} \
  -m apps.main.train \
    name=$model_name \
    seed=$seed \
    grad_acc_steps=$((256 / $tot_batch_size)) \
    steps=23842 \
    dump_dir=$save_dir \
    config=apps/main/configs/base_llama.yaml \
    data.root_dir=/datasets/llm/pretraining \
    data.batch_size=$batch_size \
    data.tokenizer.path=/datasets/pretrained-llms/Llama-3.2-1B/original/tokenizer.model \
    logging.wandb.project=lingua \
    logging.wandb.id=$model_name \
    logging.wandb.dir=$CKPT_HOME \
    model.dim=2048 \
    model.multiple_of=256 \
    model.n_layers=16 \
    model.n_heads=32 \
    model.ffn_dim_multiplier=1.5 \
    optim.lr=8e-4 \
    optim.warmup=1000 \
    optim.prune_reg_lambda=${reg_lmbdas[$SLURM_ARRAY_TASK_ID]} \
    optim.prune_config_path=apps/main/configs/prune/lasso_linear.yaml \
    optim.prune_warmup_steps=1000 \
