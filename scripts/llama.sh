#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=llama3.2-982m
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=1000G
#SBATCH --account=parq
#SBATCH --qos=parq_high
#SBATCH --array=0-5
#SBATCH --output=/fsx-parq/%u/lingua_dumps/llama3.2-982m_%A/%a/train.out
#SBATCH --error=/fsx-parq/%u/lingua_dumps/llama3.2-982m_%A/%a/train.err
#SBATCH --time=2-00:00:00
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120
#SBATCH --distribution=block

source ~/.profile
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate lingua

slurm_job_id=$SLURM_ARRAY_JOB_ID
seed=$((slurm_job_id % 1000))

head_node=`scontrol show hostnames $SLURM_JOB_NODELIST | sed 1q`
head_node_ip=`srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address`
node_port=$((29000 + seed))
echo Host node address: ${head_node_ip}:${node_port}

if [ "$CLUSTER_ID" = "fair-a100" ]; then
  lingua_tmpdir=${HOME}/lingua
else
  # Copy code snapshot to local storage
  lingua_tmpdir=${SLURM_TMPDIR:-/tmp}/lingua-$(date +%s)
  mkdir -p $lingua_tmpdir
  pushd ${HOME}/lingua
  stash_hash=$(git stash create)
  git archive --format=tar.gz ${stash_hash:-HEAD} | tar -xvzf - -C $lingua_tmpdir
  echo Saved code snapshot to $lingua_tmpdir
fi

model_name=llama3.2-982m_${slurm_job_id}
save_dir=${CKPT_HOME}/lingua_dumps/${model_name}/${SLURM_ARRAY_TASK_ID}
model_name+=_${SLURM_ARRAY_TASK_ID}
mkdir -p $save_dir

reg_lmbdas=(2e-3 5e-3 1e-2 2.5e-2 4e-2 6e-2)
tot_batch_size=128
batch_size=$((tot_batch_size / (SLURM_JOB_NUM_NODES * SLURM_GPUS_ON_NODE)))
grad_accum=$((256 / $tot_batch_size))
n_steps=23842
optim_warmup=2000
ret=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_DATASETS_OFFLINE=1 \
  srun --chdir=$lingua_tmpdir torchrun \
  --nnodes $SLURM_JOB_NUM_NODES \
  --nproc-per-node $SLURM_GPUS_ON_NODE \
  --rdzv-id $slurm_job_id \
  --rdzv-backend c10d \
  --rdzv-endpoint ${head_node_ip}:${node_port} \
  -m apps.main.train \
    name=$model_name \
    seed=$seed \
    grad_acc_steps=$grad_accum \
    steps=$n_steps \
    dump_dir=$save_dir \
    config=apps/main/configs/llama/llama3-982m.yaml \
    data.root_dir=/fsx-parq/shared/datasets \
    data.batch_size=$batch_size \
    data.tokenizer.path=${CKPT_HOME}/hub/models--meta-llama--Llama-3.1-8B/original/tokenizer.model \
    eval.validation.qos=alignment_shared \
    eval.validation.partition=learn \
    eval.validation.ncpu=${SLURM_CPUS_PER_GPU} \
    distributed.detect_anomaly=True \
    logging.wandb.project=lingua \
    logging.wandb.id=$model_name \
    logging.wandb.dir=$CKPT_HOME \
    optim.warmup=$optim_warmup \
    optim.weight_decay=0.1 \
    optim.prune_reg_lambda=${reg_lmbdas[$SLURM_ARRAY_TASK_ID]} \
    optim.prune_config_path=apps/main/configs/prune/group_attn_head.yaml \
    optim.prune_warmup_steps=$optim_warmup \
    || ret=$?

# Resubmit the job up to 3 times if it failed
restart_count=$(scontrol show job $SLURM_JOB_ID | grep -oP 'Restarts=\d+' | grep -oE '[0-9]+')
if [[ "$ret" != "0" ]] && [ $restart_count -lt 3 ]; then
  scontrol requeue $SLURM_JOB_ID
fi
