# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from copy import deepcopy
import importlib
import os
import sys
import time
from typing import Any, Iterable, Optional, Union, ContextManager

import torch
from torch._guards import active_fake_mode
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker
from torch.distributed._tools.mem_tracker import MemTracker
from torch.distributed._tools.runtime_estimator import RuntimeEstimator
import torch.multiprocessing as mp

from torch.distributed.elastic.multiprocessing.errors import record

import torchtitan.components.ft as ft
import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger


class FakeTrainer(torch.distributed.checkpoint.stateful.Stateful):
    job_config: JobConfig
    gc_handler: utils.GarbageCollection

    parallel_dims: ParallelDims
    train_spec: train_spec_module.TrainSpec
    world_mesh: torch.distributed.DeviceMesh

    model_parts: list[torch.nn.Module]
    optimizers: train_spec_module.OptimizersContainer
    lr_schedulers: train_spec_module.LRSchedulersContainer

    pp_has_first_stage: bool
    pp_has_last_stage: bool

    device: torch.device

    # states
    step: int

    # vocab size
    vocab_size: int

    # estimator
    estimator: Union[MemTracker, FSDPMemTracker, RuntimeEstimator]
    init_mode: ContextManager

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, rank: int, job_config: JobConfig):
        self.job_config = job_config
        job_config.training.steps = 2

        if job_config.memory_estimation.enabled and job_config.runtime_estimation.enabled:
            logger.info(
                f"Cannot estimate memory and runtime together."
                f"Enable only one at a time."
            )

        if job_config.estimation.disable_fake_mode:
            self.init_mode = nullcontext()
        else:
            self.init_mode = FakeTensorMode(allow_non_fake_inputs=True)


        if job_config.fault_tolerance.enable:
            job_config.fault_tolerance.enable = False
            logger.info(
                f"Disabling fault tolerance under estimation mode."
                f" Support will be added later."
            )

        if job_config.model.norm_type == "compiled_rmsnorm":
            logger.info("Compiled RMSNorm is not supported yet. Switching to RMSNorm.")
            job_config.model.norm_type = "rmsnorm"

        if job_config.training.compile or job_config.parallelism.enable_compiled_autograd:
            logger.info("Compile mode is not supported yet. Switching to eager mode.")
            job_config.training.compile = False
            job_config.parallelism.enable_compiled_autograd = False

        logger.info(f"Starting job: {job_config.job.description}")

        if job_config.experimental.custom_import:
            importlib.import_module(job_config.experimental.custom_import)

        if job_config.job.print_args:
            logger.info(f"Running with args: {job_config.to_dict()}")

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        device_module.set_device(self.device)
        ft_manager = ft.init_ft_manager(job_config)

        # init distributed
        world_size = int(os.environ["WORLD_SIZE"])

        parallelism_config = job_config.parallelism
        self.parallel_dims = parallel_dims = ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=not parallelism_config.disable_loss_parallel,
        )

        # init fake pg
        store = FakeStore()
        torch.distributed.init_process_group(
            "fake", rank=rank, world_size=world_size, store=store, group_name="custom"
        )

        # build meshes
        self.world_mesh = world_mesh = parallel_dims.build_mesh(device_type=device_type)
        if parallel_dims.dp_enabled:
            dp_mesh = world_mesh["dp"]
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0

        self.train_spec = train_spec_module.get_train_spec(job_config.model.name)

        # init tokenizer
        tokenizer = self.train_spec.build_tokenizer_fn(job_config)
        self.vocab_size = tokenizer.n_words

        # build model (using meta init)
        model_cls = self.train_spec.cls
        model_args = self.train_spec.config[job_config.model.flavor]
        # set the model args from training job configs
        model_args.update_from_config(job_config, tokenizer)
        if model_args.vocab_size == -1:
            model_args.vocab_size = self.vocab_size

        logger.info(
            f"Building {self.train_spec.name} {job_config.model.flavor} with {model_args}"
        )
        with torch.device("meta"):
            model = model_cls.from_model_args(model_args)

        # Build the collection of model converters. No-op if `model.converters` empty
        model_converters = build_model_converters(job_config, parallel_dims)
        model_converters.convert(model)

        # log model size
        model_param_count = utils.get_num_params(model)

        logger.info(
            f"Model {self.train_spec.name} {job_config.model.flavor} "
            f"size: {model_param_count:,} total parameters"
        )

        # move sharded model to CPU/GPU and initialize weights via DTensor
        if job_config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = device_type
        else:
            init_device = device_type
            buffer_device = None

        # apply parallelisms and initialization
        if parallel_dims.pp_enabled:
            if not self.train_spec.pipelining_fn:
                raise RuntimeError(
                    f"Pipeline Parallel is enabled but {self.train_spec.name} "
                    f"does not support pipelining"
                )

            # apply both PT-D Pipeline Parallel and SPMD-style PT-D techniques
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = self.train_spec.pipelining_fn(
                model,
                world_mesh,
                parallel_dims,
                job_config,
                self.device,
                model_args,
                self.train_spec.parallelize_fn,
                self.train_spec.loss_fn,
            )
            # when PP is enabled, `model` obj is no longer used after this point,
            # model_parts is used instead
            del model
            with self.init_mode:
                for m in self.model_parts:
                    m.to_empty(device=init_device)
                    if not active_fake_mode():
                        with torch.no_grad():
                            m.init_weights(buffer_device=buffer_device)
                    m.train()
        else:
            # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
            model = self.train_spec.parallelize_fn(
                model, world_mesh, parallel_dims, job_config
            )
            with self.init_mode:
                model.to_empty(device=init_device)
                if not active_fake_mode():
                    with torch.no_grad():
                        model.init_weights(buffer_device=buffer_device)
                model.train()

            self.model_parts = [model]

        with self.init_mode:
            # build optimizer after applying parallelisms to the model
            self.optimizers = self.train_spec.build_optimizers_fn(
                self.model_parts, job_config, ft_manager
            )
            self.lr_schedulers = self.train_spec.build_lr_schedulers_fn(
                self.optimizers, job_config
            )
            # Post optimizer step model converters hook.
            # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
            # where it issues a single all-reduce for all parameters at once for better performance
            self.optimizers.register_step_post_hook(
                lambda *args, **kwargs: model_converters.post_optimizer_hook(
                    self.model_parts
                )
            )

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0
        self.train_context = dist_utils.get_train_context(
            parallel_dims.loss_parallel_enabled,
            parallelism_config.enable_compiled_autograd,
        )

        logger.info(
            "Trainer initialized. "
            f"Training starts at step {self.step + 1}, "
            f"with local batch size {job_config.training.batch_size}, "
            f"global batch size {job_config.training.batch_size * dp_degree}, "
            f"sequence length {job_config.training.seq_len}, "
            f"total steps {job_config.training.steps} "
            f"(warmup {job_config.lr_scheduler.warmup_steps})"
        )

    def next_batch(self,) -> tuple[torch.Tensor, torch.Tensor]:
        with self.init_mode:
            device_type = utils.device_type
            batch = (
                torch.randint(
                    0,
                    self.vocab_size,
                    (self.job_config.training.batch_size, self.job_config.training.seq_len),
                    device=device_type,
                ),
                torch.randint(
                    0,
                    self.vocab_size,
                    (self.job_config.training.batch_size, self.job_config.training.seq_len),
                    device=device_type,
                ),
            )
            input_ids, labels = batch
        return input_ids, labels

    def train_step(self, inputs: torch.Tensor, labels: torch.Tensor):
        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        model_parts = self.model_parts
        world_mesh = self.world_mesh
        parallel_dims = self.parallel_dims
        cp_mesh = None
        pp_mesh = None
        if parallel_dims.cp_enabled:
            cp_mesh = world_mesh["cp"]
            pp_mesh = world_mesh["pp"]

        with self.init_mode:
            # apply context parallelism if cp is enabled
            # ensure CP handles the separate freqs_cis buffer for each pp stage
            optional_context_parallel_ctx = (
                dist_utils.create_context_parallel_ctx(
                    cp_mesh=cp_mesh,
                    cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
                    cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                    cp_no_restore_buffers={inputs, labels},
                    cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
                )
                if parallel_dims.cp_enabled
                else None
            )
            if parallel_dims.pp_enabled:
                # Pipeline Parallel forward / backward inside step() call
                with self.train_context(optional_context_parallel_ctx):
                    targets, losses = (
                        (labels, []) if self.pp_has_last_stage else (None, None)
                    )
                    if self.pp_has_first_stage:
                        self.pp_schedule.step(inputs, target=targets, losses=losses)
                    else:
                        self.pp_schedule.step(target=targets, losses=losses)

                # accumulate losses across pipeline microbatches
                # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
                loss = (
                    torch.mean(torch.stack(losses)).to(self.device)
                    if self.pp_has_last_stage
                    else torch.tensor([-1.0], device=self.device)
                )
            else:
                # Non-PP forward / backward
                with self.train_context(optional_context_parallel_ctx):
                    assert len(model_parts) == 1
                    pred = model_parts[0](inputs)
                    loss = self.train_spec.loss_fn(pred, labels)
                    # pred.shape=(bs, seq_len, vocab_size)
                    # need to free to before bwd to avoid peaking memory
                    del pred
                    loss.backward()

            dist_utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters()],
                self.job_config.training.max_norm,
                foreach=True,
                pp_mesh=pp_mesh,
            )
            self.optimizers.step()
            # self.lr_schedulers.step()
            self.optimizers.zero_grad()

    @record
    def train(self):
        job_config = self.job_config
        assert job_config.memory_estimation.enabled or job_config.runtime_estimation.enabled, (
            f"One of the estimations (memory_estimation/runtime_estimation) must be enabled."
        )
        
        if job_config.memory_estimation.enabled:
            if self.parallel_dims.dp_shard_enabled:
                # TODO: Handle multiple model parts
                self.estimator = FSDPMemTracker(mod=self.model_parts[0], optm=self.optimizers.optimizers[0])
            else:
                self.estimator = MemTracker()
                self.estimator.track_external(self.model_parts, self.optimizers.optimizers)
        else:
            self.estimator = RuntimeEstimator(self.world_mesh["pp"].get_local_rank())
            self.estimator.fake_mode = self.init_mode
        with self.estimator("operator-level-cost-model") if job_config.runtime_estimation.enabled else self.estimator:
            while self.step < job_config.training.steps:
                self.step += 1
                self.gc_handler.run(self.step)
                inputs, labels = self.next_batch()
                self.train_step(inputs, labels)
                if self.step == 1 and job_config.memory_estimation.enabled:
                    self.estimator.reset_mod_stats()  # iter 0 does not have optimizer state

        if job_config.memory_estimation.enabled:
            self.estimator.display_modulewise_snapshots(depth=3, units="MiB", tabulate=True)
        else:
            self.estimator.display_modulewise_stats(depth=3)

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)
        logger.info("Training completed")


def estimate(pp_rank: int, pp_size: int, args_list: list):
    config = JobConfig()
    config.maybe_add_custom_args()
    config.parse_args(args_list=args_list)
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size % pp_size == 0, "world size should be divisble by pipeline parallel size"
    spmd_size = world_size // pp_size
    rank = (pp_rank * spmd_size)
    init_logger()
    trainer: Optional[FakeTrainer] = None

    try:
        trainer = FakeTrainer(rank, config)
        trainer.train()
    finally:

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            logger.info("Process group destroyed.")

if __name__ == "__main__":
    args_list: list = sys.argv[1:]
    config = JobConfig()
    config.maybe_add_custom_args()
    config.parse_args()

    pp_size = config.parallelism.pipeline_parallel_degree
    try: 
        for pp_rank in range(pp_size):
            estimate(pp_size=pp_size, pp_rank=pp_rank, args_list=deepcopy(args_list))
        # mp.spawn(estimate, args=(pp_size, args_list), nprocs=pp_size, join=True)
    except Exception as e:
        print(e)
        print("Unsuccessful in launching multiple processes")


    