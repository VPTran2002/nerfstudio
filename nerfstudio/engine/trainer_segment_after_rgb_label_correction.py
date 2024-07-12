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
import numpy as np
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
class TrainerSegmentAfterRGBLabelCorrectionConfig(ExperimentSegmentConfig):
    """Configuration for training regimen"""

    _target: Type = field(default_factory=lambda: TrainerSegmentAfterRGBLabelCorrection)
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


class TrainerSegmentAfterRGBLabelCorrection:
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

    def __init__(self, config: TrainerSegmentAfterRGBLabelCorrectionConfig, local_rank: int = 0, world_size: int = 1) -> None:
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

        self.callbacks = self.pipeline.get_training_callbacks( # [step_cb,after_train,refinement_after]
            TrainingCallbackAttributes(
                optimizers=self.optimizers, grad_scaler=self.grad_scaler, pipeline=self.pipeline, trainer=self
            )
        )

        # set up writers/profilers if enabled
        writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        writer.setup_event_writer(
            self.config.is_wandb_enabled(),
            self.config.is_tensorboard_enabled(),
            self.config.is_comet_enabled(),
            log_dir=writer_log_path,
            experiment_name=self.config.experiment_name,
            project_name=self.config.project_name,
        )
        writer.setup_local_writer(
            self.config.logging, max_iter=self.config.max_num_iterations, banner_messages=banner_messages
        )
        writer.put_config(name="config", config_dict=dataclasses.asdict(self.config), step=0)
        profiler.setup_profiler(self.config.logging, writer_log_path)

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

    def train(self) -> None:
        """Train the model."""
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"

        self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform( #not so important
            self.base_dir / "dataparser_transforms.json"
        )

        self._init_viewer_state()
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.max_num_iterations #30000
            step = 0
            self.stop_training = False
            for step in range(self._start_step, self._start_step + num_iterations): #train for 30000 iterations
                self.step = step
                if self.stop_training:
                    break
                while self.training_state == "paused":
                    if self.stop_training:
                        self._after_train()
                        return
                    time.sleep(0.01)
                with self.train_lock: #Not relevant since we do not do multiprocessing
                    with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                        self.pipeline.train() #set pipeline into training mode

                        # training callbacks before the training iteration
                        for callback in self.callbacks: #run all callbacks which are supposed to be runned before training iteration
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                            )

                        # time the forward pass
                        loss, loss_dict, metrics_dict = self.train_iteration(step) #do one training iteration

                        # training callbacks after the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                            )

                # Skip the first two steps to avoid skewed timings that break the viewer rendering speed estimate.
                if step > 1:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC,
                        duration=self.world_size
                        * self.pipeline.datamanager.get_train_rays_per_batch()
                        / max(0.001, train_t.duration),
                        step=step,
                        avg_over_steps=True,
                    )

                self._update_viewer_state(step) #renders scene using pipeline and measures how long rendering takes

                # a batch of train rays
                if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                    writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                    writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                    writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)
                    # The actual memory allocated by Pytorch. This is likely less than the amount
                    # shown in nvidia-smi since some unused memory can be held by the caching
                    # allocator and some context needs to be created on GPU. See Memory management
                    # (https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
                    # for more details about GPU memory management.
                    writer.put_scalar(
                        name="GPU Memory (MB)", scalar=torch.cuda.max_memory_allocated() / (1024**2), step=step
                    )

                # Do not perform evaluation if there are no validation images
                if self.pipeline.datamanager.eval_dataset:
                    self.eval_iteration(step)

                if step_check(step, self.config.steps_per_save):
                    self.save_checkpoint(step)

                writer.write_out_storage()

        # save checkpoint at the end of training, and write out any remaining events
        self._after_train()

    def shutdown(self) -> None:
        """Stop the trainer and stop all associated threads/processes (such as the viewer)."""
        self.stop_training = True  # tell the training loop to stop
        if self.viewer_state is not None:
            # stop the viewer
            # this condition excludes the case where `viser_server` is either `None` or an
            # instance of `viewer_legacy`'s `ViserServer` instead of the upstream one.
            if isinstance(self.viewer_state.viser_server, viser.ViserServer):
                self.viewer_state.viser_server.stop()

    def _after_train(self) -> None:
        """Function to run after training is complete"""
        self.training_state = "completed"  # used to update the webui state
        # save checkpoint at the end of training
        self.save_checkpoint(self.step)
        # write out any remaining events (e.g., total train time)
        writer.write_out_storage()
        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("Config File", str(self.config.get_base_dir() / "config.yml"))
        table.add_row("Checkpoint Directory", str(self.checkpoint_dir))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Training Finished :tada:[/bold]", expand=False))

        # after train end callbacks
        for callback in self.callbacks:
            callback.run_callback_at_location(step=self.step, location=TrainingCallbackLocation.AFTER_TRAIN)

        if not self.config.viewer.quit_on_train_completion:
            self._train_complete_viewer()

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

    @check_viewer_enabled
    def _update_viewer_state(self, step: int) -> None:
        """Updates the viewer state by rendering out scene with current pipeline
        Returns the time taken to render scene.

        Args:
            step: current train step
        """
        assert self.viewer_state is not None
        num_rays_per_batch: int = self.pipeline.datamanager.get_train_rays_per_batch()
        try:
            self.viewer_state.update_scene(step, num_rays_per_batch)
        except RuntimeError:
            time.sleep(0.03)  # sleep to allow buffer to reset
            CONSOLE.log("Viewer failed. Continuing training.")

    @check_viewer_enabled
    def _train_complete_viewer(self) -> None:
        """Let the viewer know that the training is complete"""
        assert self.viewer_state is not None
        self.training_state = "completed"
        try:
            self.viewer_state.training_complete()
        except RuntimeError:
            time.sleep(0.03)  # sleep to allow buffer to reset
            CONSOLE.log("Viewer failed. Continuing training.")
        CONSOLE.print("Use ctrl+c to quit", justify="center")
        while True:
            time.sleep(0.01)

    @check_viewer_enabled
    def _update_viewer_rays_per_sec(self, train_t: TimeWriter, vis_t: TimeWriter, step: int) -> None:
        """Performs update on rays/sec calculation for training

        Args:
            train_t: timer object carrying time to execute total training iteration
            vis_t: timer object carrying time to execute visualization step
            step: current step
        """
        train_num_rays_per_batch: int = self.pipeline.datamanager.get_train_rays_per_batch()
        writer.put_time(
            name=EventName.TRAIN_RAYS_PER_SEC,
            duration=self.world_size * train_num_rays_per_batch / (train_t.duration - vis_t.duration),
            step=step,
            avg_over_steps=True,
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

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers

        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path: Path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else self.pipeline.state_dict(),
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "schedulers": {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("*"):
                if f != ckpt_path:
                    f.unlink()

    def find_overlapping_instances(self, M1, M2, mappper):
        # Get unique instances from M1 and M2, excluding the background (assuming background is labeled as 0)
        unique_M1 = np.unique(M1)
        unique_M1 = unique_M1[unique_M1 != 0]

        # Iterate over each unique instance in M1
        for instance in unique_M1:
            # Create a mask for the current instance in M1
            mask_M1 = (M1 == instance)

            # Find overlapping instances in M2
            overlapping_instances, counts = np.unique(M2[mask_M1], return_counts=True)
            
            # Exclude background (assuming background is labeled as 0)
            total = counts.sum()
            valid_idx = np.logical_and(overlapping_instances != 0, counts/total > 0.2)
            valid_overlapping_instances = tuple(sorted(overlapping_instances[valid_idx]))
            valid_counts = counts[valid_idx]
            if valid_counts.size > 0 and not(valid_overlapping_instances in mappper):
                max_count_idx = np.argmax(valid_counts)
                mappper[valid_overlapping_instances] = valid_overlapping_instances[max_count_idx]

    def create_consistent_images(self, M1, M2, mappper):
        # Get unique instances from M1 and M2, excluding the background (assuming background is labeled as 0)
        # Get unique instances from M1 and M2, excluding the background (assuming background is labeled as 0)
        unique_M1 = np.unique(M1)
        unique_M1 = unique_M1[unique_M1 != 0]

        result_mask = np.zeros_like(M1)
        # Iterate over each unique instance in M1
        for instance in unique_M1:
            # Create a mask for the current instance in M1
            mask_M1 = (M1 == instance)

            # Find overlapping instances in M2
            overlapping_instances, counts = np.unique(M2[mask_M1], return_counts=True)
            
            # Exclude background (assuming background is labeled as 0)
            total = counts.sum()
            valid_idx = np.logical_and(overlapping_instances != 0, counts/total > 0.2)
            valid_overlapping_instances = tuple(sorted(overlapping_instances[valid_idx]))
            valid_counts = counts[valid_idx]
            if valid_counts.size > 0:
                result_mask[mask_M1] = mappper[valid_overlapping_instances]
    
        return result_mask                

    def label_correction(self):
        self.pipeline.eval()
        mapper = {}
        cameras = self.pipeline.datamanager.train_dataset.cameras #type: ignore
        for i in range(len(cameras)):
            mask_consistent = self.pipeline.datamanager.cached_train[i]["image_inst_segm"].squeeze()
            camera = cameras[i:i+1].to(torch.device("cuda:0"))
            output = self.pipeline.model.get_outputs(camera)
            mask=torch.argmax(output["instance_segmentation_logits"], dim=2) # type: ignore
            self.find_overlapping_instances(mask_consistent.to(torch.device("cpu")).numpy(), mask.to(torch.device("cpu")).numpy(), mapper)

        for i in range(len(cameras)):
            mask_consistent = self.pipeline.datamanager.cached_train[i]["image_inst_segm"].squeeze()
            camera = cameras[i:i+1].to(torch.device("cuda:0"))
            output = self.pipeline.model.get_outputs(camera)
            mask=torch.argmax(output["instance_segmentation_logits"], dim=2) # type: ignore
            new_mask = self.create_consistent_images(mask_consistent.to(torch.device("cpu")).numpy(), mask.to(torch.device("cpu")).numpy(), mapper)
            self.pipeline.datamanager.cached_train[i]["image_inst_segm"] = torch.from_numpy(new_mask).squeeze()
        self.pipeline.train()

    @profiler.time_function
    def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """
        if self.pipeline.config.model.rgb_train_length < step and step%self.pipeline.config.model.label_correction_every==0: # type: ignore
            print("Label correction...")
            self.label_correction()

        needs_zero = [
            group for group in self.optimizers.parameters.keys() if step % self.gradient_accumulation_steps[group] == 0
        ] #For which groups do we needto call zero_grad?
        self.optimizers.zero_grad_some(needs_zero) #do zero grad for all groups which need it
        cpu_or_cuda_str: str = self.device.split(":")[0] #cuda
        cpu_or_cuda_str = "cpu" if cpu_or_cuda_str == "mps" else cpu_or_cuda_str #cuda

        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision): #get training loss of current iteration
            _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step) 
            loss = functools.reduce(torch.add, loss_dict.values())
        self.grad_scaler.scale(loss).backward()  # type: ignore
        if self.pipeline.config.model.rgb_train_length >= step: # type: ignore
            needs_step = [ #which parameter groups require optimization step now?
                group
                for group in self.optimizers.parameters.keys()
                if step % self.gradient_accumulation_steps[group] == self.gradient_accumulation_steps[group] - 1
            ]
        else:
            needs_step = [
                group 
                for group in self.optimizers.parameters.keys()
                if (step % self.gradient_accumulation_steps[group] == 0 and (group == "upproj" or group == "features_segmentation_small"))
            ]
        self.optimizers.optimizer_scaler_step_some(self.grad_scaler, needs_step) #Do an optimization step for those which require...

        if self.config.log_gradients: #We do not step in here
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad  # type: ignore
                    total_grad += grad

            metrics_dict["Gradients/Total"] = cast(torch.Tensor, total_grad)  # type: ignore

        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale():
            self.optimizers.scheduler_step_all(step) #schedule learning rate

        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict  # type: ignore

    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step: int) -> None:
        """Run one iteration with different batch/image/all image evaluations depending on step size.

        Args:
            step: Current training step.
        """
        # a batch of eval rays
        if step_check(step, self.config.steps_per_eval_batch):
            _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(step=step)
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())
            writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=step)
            writer.put_dict(name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=step)
            writer.put_dict(name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step)

        # one eval image
        if step_check(step, self.config.steps_per_eval_image):
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                metrics_dict, images_dict = self.pipeline.get_eval_image_metrics_and_images(step=step)
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=step,
                avg_over_steps=True,
            )
            writer.put_dict(name="Eval Images Metrics", scalar_dict=metrics_dict, step=step)
            group = "Eval Images"
            for image_name, image in images_dict.items():
                writer.put_image(name=group + "/" + image_name, image=image, step=step)

        # all eval images
        if step_check(step, self.config.steps_per_eval_all_images):
            metrics_dict = self.pipeline.get_average_eval_image_metrics(step=step)
            writer.put_dict(name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dict, step=step)
