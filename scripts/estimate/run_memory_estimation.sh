#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overrides for convenience
# e.g.
# NGPU=4 ./run_memory_estimation.sh
NGPU=${NGPU:-"4"}
NNODES=${NNODES:-"1"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

CONFIG_FILE=${CONFIG_FILE:-"./train_configs/debug_model.toml"}

# BATCH_SIZES=(2 2 2 1 1)
BATCH_SIZES=(8 8 8 8 16 16 16 16)
SEQ_LENS=(1024 1024 2048 2048 1024 1024 2048 2048)
NUM_STAGES=(4 8 4 8 4 8 4 8)
DUMP_FOLDER=/n/netscratch/idreos_lab/Lab/sboulware/torchtitan/outputs/debug_model_PP_estimation
SCHEDULES=(GPipe 1F1B Interleaved1F1B LoopedBFS)

# Calculate WORLD_SIZE as the product of NGPU and NNODES
# Export WORLD_SIZE and LOCAL_RANK
export WORLD_SIZE=$((NGPU * NNODES))
export RANK=0
export LOCAL_RANK=0

for ((j=0; j<4; j++)); do
    SCHEDULE=${SCHEDULES[j]}
    for ((i=0; i<8; i++)); do
        BATCH_SIZE=${BATCH_SIZES[i]}
        SEQ_LEN=${SEQ_LENS[i]}
        NUM_STAGES=${NUM_STAGES[i]}

        python estimation.py \
        --job.config_file ${CONFIG_FILE} \
        --job.dump_folder ${DUMP_FOLDER} \
        --training.batch_size $BATCH_SIZE \
        --training.seq_len $SEQ_LEN \
        --parallelism.pipeline_parallel_degree $NUM_STAGES \
        --parallelism.pipeline_parallel_microbatches $BATCH_SIZE \
        --parallelism.pipeline_parallel_schedule $SCHEDULE \
        --memory_estimation.enabled \
        $overrides
    
    done
done

