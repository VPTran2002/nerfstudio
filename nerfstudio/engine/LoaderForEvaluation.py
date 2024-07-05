# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code to train model.
"""
from __future__ import annotations

import dataclasses
import functools
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import DefaultDict, Dict, List, Literal, Optional, Tuple, Type, cast

import torch
import viser
from rich import box, style
from rich.panel import Panel
from rich.table import Table
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.configs.experiment_config import ExperimentSegmentConfig
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.pipelines.base_pipeline import VanillaPipelineSegment
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import check_eval_enabled, check_main_thread, check_viewer_enabled
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.viewer.viewer import Viewer as ViewerState
from nerfstudio.viewer_legacy.server.viewer_state import ViewerLegacyState

TRAIN_INTERATION_OUTPUT = Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
TORCH_DEVICE = str


@dataclass
class TrainerSegmentAfterRGBOnlyLoadConfig(ExperimentSegmentConfig):
    """Configuration for training regimen"""

    _target: Type = field(default_factory=lambda: TrainerSegmentAfterRGBOnlyLoad)
    """target class to instantiate"""
    steps_per_save: int = 1000
    """Number of steps between saves."""
    steps_per_eval_batch: int = 500
    """Number of steps between randomly sampled batches of rays."""
    steps_per_eval_image: int = 500
    """Number of steps between single eval images."""
    steps_per_eval_all_images: int = 25000
    """Number of steps between eval all images."""
    max_num_iterations: int = 1000000
    """Maximum number of iterations to run."""
    mixed_precision: bool = False
    """Whether or not to use mixed precision for training."""
    use_grad_scaler: bool = False
    """Use gradient scaler even if the automatic mixed precision is disabled."""
    save_only_latest_checkpoint: bool = True
    """Whether to only save the latest checkpoint or all checkpoints."""
    # optional parameters if we want to resume training
    load_dir: Optional[Path] = None
    """Optionally specify a pre-trained model directory to load from."""
    load_step: Optional[int] = None
    """Optionally specify model step to load from; if none, will find most recent model in load_dir."""
    load_config: Optional[Path] = None
    """Path to config YAML file."""
    load_checkpoint: Optional[Path] = None
    """Path to checkpoint file."""
    log_gradients: bool = False
    """Optionally log gradients during training"""
    gradient_accumulation_steps: Dict[str, int] = field(default_factory=lambda: {})
    """Number of steps to accumulate gradients over. Contains a mapping of {param_group:num}"""


class TrainerSegmentAfterRGBOnlyLoad:
    """Trainer class

    Args:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.

    Attributes:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.
        device: The device to run the training on.
        pipeline: The pipeline object.
        optimizers: The optimizers object.
        callbacks: The callbacks object.
        training_state: Current model training state.
    """

    pipeline: VanillaPipelineSegment
    optimizers: Optimizers
    callbacks: List[TrainingCallback]

    def __init__(self, config: TrainerSegmentAfterRGBOnlyLoadConfig, local_rank: int = 0, world_size: int = 1) -> None:
        self.train_lock = Lock() #Lock for threads
        self.config = config #Training configuration for splatfacto model
        self.local_rank = local_rank #This is 0 (since we use only one process)
        self.world_size = world_size #1 since we only use 1 gpu
        self.device: TORCH_DEVICE = config.machine.device_type #cuda
        if self.device == "cuda":
            self.device += f":{local_rank}" #every process uses exactly one gpu
        self.mixed_precision: bool = self.config.mixed_precision #False: Do not train with mixed precision
        self.use_grad_scaler: bool = self.mixed_precision or self.config.use_grad_scaler #We do not train using a grad_scaler since we did not specify this explicitely and do not train with mixed precision
        self.training_state: Literal["training", "paused", "completed"] = "training" #The trainer is initialized to be in training state first
        self.gradient_accumulation_steps: DefaultDict = defaultdict(lambda: 1)
        self.gradient_accumulation_steps.update(self.config.gradient_accumulation_steps) #if we try to access a key of gradient_accumulation_step, we get 1 (default) back as we do not use gradient accumulation

        if self.device == "cpu":
            self.mixed_precision = False
            CONSOLE.print("Mixed precision is disabled for CPU training.")
        self._start_step: int = 0
        # optimizers
        self.grad_scaler = GradScaler(enabled=self.use_grad_scaler) #The grad scaler acts like an identity function since use_grad_scaler is deactivated

        self.base_dir: Path = config.get_base_dir() #here, checkpoints, config files and everything else is stored
        # directory to save checkpoints
        self.checkpoint_dir: Path = config.get_checkpoint_dir() #here the model is saved. It is a subdirectory of base_dir
        CONSOLE.log(f"Saving checkpoints to: {self.checkpoint_dir}")

        self.viewer_state = None

        # used to keep track of the current step
        self.step = 0

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datasets into memory
                'inference': does not load any dataset into memory
        """
        self.pipeline = self.config.pipeline.setup( #The pipeline is setup. Here dataloading and the model come together. Later, for example, one only has to call self.pipeline.training_step to load data, put it through model
            device=self.device, #cuda:0
            test_mode=test_mode, #'val'
            world_size=self.world_size,#1
            local_rank=self.local_rank,#0
            grad_scaler=self.grad_scaler,#identity
        )
        self.optimizers = self.setup_optimizers() #Here the optimizers for every param group are set up

        # set up viewer if enabled
        viewer_log_path = self.base_dir / self.config.viewer.relative_log_filename #Where is the log file for the viewer? 'outputs/unnamed/splatfacto_segment/2024-05-20_104454/viewer_log_filename.txt'
        self.viewer_state, banner_messages = None, None #None, None
        if self.config.is_viewer_legacy_enabled() and self.local_rank == 0: #This is disabled
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = ViewerLegacyState(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
            )
            banner_messages = [f"Legacy viewer at: {self.viewer_state.viewer_url}"]
        if self.config.is_viewer_enabled() and self.local_rank == 0:
            datapath = self.config.data #None
            if datapath is None:
                datapath = self.base_dir # 'outputs/unnamed/splatfacto_segment/2024-05-20_104454'
            self.viewer_state = ViewerState(
                self.config.viewer, #ViewerConfig
                log_filename=viewer_log_path, # 'outputs/unnamed/splatfacto_segment/2024-05-20_104454/viewer_log_filename.txt'
                datapath=datapath, # 'outputs/unnamed/splatfacto_segment/2024-05-20_104454'
                pipeline=self.pipeline, #splatfacto_segment pipeline
                trainer=self, #this instance
                train_lock=self.train_lock, #The train lock
                share=self.config.viewer.make_share_url, # False
            )
            banner_messages = self.viewer_state.viewer_info
        self._check_viewer_warnings()

        self._load_checkpoint()

    def setup_optimizers(self) -> Optimizers:
        """Helper to set up the optimizers

        Returns:
            The optimizers object given the trainer config.
        """
        #What are parameter groups? All parameter of the model are grouped resulting in parameter groups. For example, the parameter groups
        #of splatfacto are means (50000,3), features_dc (50000,3), features_rest (50000, 15, 3), opacities (50000, 1), scales (50000, 3), quats (50000, 4).
        #Note that scales and quats form the covariance matrix
        optimizer_config = self.config.optimizers.copy() #Here, it is configured which optimizer is used for which parameter group
        param_groups = self.pipeline.get_param_groups() #Here, we get everything apart from camera_opt
        return Optimizers(optimizer_config, param_groups)


    @check_main_thread
    def _check_viewer_warnings(self) -> None:
        """Helper to print out any warnings regarding the way the viewer/loggers are enabled"""
        if (
            (self.config.is_viewer_legacy_enabled() or self.config.is_viewer_enabled())
            and not self.config.is_tensorboard_enabled()
            and not self.config.is_wandb_enabled()
            and not self.config.is_comet_enabled()
        ):
            string: str = (
                "[NOTE] Not running eval iterations since only viewer is enabled.\n"
                "Use [yellow]--vis {wandb, tensorboard, viewer+wandb, viewer+tensorboard}[/yellow] to run with eval."
            )
            CONSOLE.print(f"{string}")

    @check_viewer_enabled
    def _init_viewer_state(self) -> None:
        """Initializes viewer scene with given train dataset"""
        assert self.viewer_state and self.pipeline.datamanager.train_dataset
        self.viewer_state.init_scene(
            train_dataset=self.pipeline.datamanager.train_dataset,
            train_state="training",
            eval_dataset=self.pipeline.datamanager.eval_dataset,
        )

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.load_dir
        load_checkpoint = self.config.load_checkpoint
        if load_dir is not None:
            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest Nerfstudio checkpoint from load_dir...")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_path}")
        elif load_checkpoint is not None:
            assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
            loaded_state = torch.load(load_checkpoint, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_checkpoint}")
        else:
            CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")
