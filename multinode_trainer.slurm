#!/bin/bash

#SBATCH --job-name=torchtitan
#SBATCH --partition=seas_gpu,gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --time=00:15:00
#SBATCH --mem=64gb
#SBATCH --output=/n/netscratch/idreos_lab/Lab/sboulware/torchtitan/job_logs/%x_%j.out
#SBATCH --error=/n/netscratch/idreos_lab/Lab/sboulware/torchtitan/job_logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --chdir=/n/netscratch/idreos_lab/Lab/sboulware/torchtitan

scontrol show job $SLURM_JOB_ID

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME="ib0,em4"

GPUS_PER_NODE=4
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "SLURM_NNODES: $SLURM_NNODES"

CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama/train_configs/debug_model.toml"}

LOG_RANK=${LOG_RANK:-0}

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"${head_node_ip}:29510"}

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

srun --output job_logs/%x_%j_%t.out --error job_logs/%x_%j_%t.err torchrun \
      --nnodes ${SLURM_NNODES} \
      --nproc_per_node ${GPUS_PER_NODE} \
      --rdzv_id 101 \
      --rdzv_backend c10d \
      --rdzv_endpoint "${head_node_ip}:29500" \
      --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
      -m torchtitan.train --job.config_file ${CONFIG_FILE}
 
