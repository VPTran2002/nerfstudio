# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Data manager that outputs cameras / images instead of raybundles

Good for things like gaussian splatting which require full cameras instead of the standard ray
paradigm
"""

from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, ForwardRef, Generic, List, Literal, Optional, Tuple, Type, Union, cast, get_args, get_origin

import cv2
import matplotlib.cm
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

from rich.progress import track
from torch.nn import Parameter
import tqdm
from typing_extensions import assert_never

from nerfstudio.cameras.camera_utils import fisheye624_project, fisheye624_unproject_helper
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, TSegmentDataset
from nerfstudio.data.dataparsers.base_dataparser import DataparserInstSegmOutputs
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset, InputDatasetSkipped
from nerfstudio.utils.misc import get_orig_class
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.viewselection.SimpleViewSelection import SimpleViewSelection
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pickle
import gc
import concurrent.futures



@dataclass
class FullImageInstanceSegmentationDatamanagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: FullImageInstanceSegmentationDatamanager)
    dataparser: AnnotatedDataParserUnion = field(default_factory=NerfstudioDataParserConfig)
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    eval_num_images_to_sample_from: int = -1
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = -1
    """When not evaluating on all images, number of iterations before picking
    new images. If -1, never pick new images."""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the image indices to use during eval; if None, uses all."""
    cache_images: Literal["cpu", "gpu"] = "cpu"
    """Whether to cache images in memory. If "cpu", caches on cpu. If "gpu", caches on device."""
    cache_images_type: Literal["uint8", "float32"] = "float32"
    """The image type returned from manager, caching images in uint8 saves memory"""
    max_thread_workers: Optional[int] = None
    """The maximum number of threads to use for caching images. If None, uses all available threads."""


class FullImageInstanceSegmentationDatamanager(DataManager, Generic[TSegmentDataset]):
    """
    A datamanager that outputs full images and cameras instead of raybundles. This makes the
    datamanager more lightweight since we don't have to do generate rays. Useful for full-image
    training e.g. rasterization pipelines
    """

    config: FullImageInstanceSegmentationDatamanagerConfig
    train_dataset: TSegmentDataset
    eval_dataset: TSegmentDataset
    map_id_color: dict
    all_ids: list

    def __init__(
        self,
        config: FullImageInstanceSegmentationDatamanagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config #configuration of FullImageDatamanagerConfig
        self.undistort_inst_seg_masks = self.config.dataparser.undistort_inst_seg_masks # type: ignore #True
        self.device = device #cuda:0
        self.world_size = world_size #1
        self.local_rank = local_rank #0
        self.sampler = None #None
        self.test_mode = test_mode #val
        self.test_split = "test" if test_mode in ["test", "inference"] else "val" #val
        self.dataparser_config = self.config.dataparser #scannetpp_dataparser
        if self.config.data is not None: #we do not go in here
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data # '/usr/stud/tranv/storage/tranv/Research/OOD/BE5DW/scannetpp/DownloadedScenesVPNouri/data/scene_id'
        self.dataparser = self.dataparser_config.setup() #initialize the dataparser
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time

        self.train_dataparser_outputs: DataparserInstSegmOutputs = self.dataparser.get_dataparser_outputs(split="train") #parser for training data
        self.train_dataset = self.create_train_dataset() #This is the training InputDataset
        self.eval_dataset = self.create_eval_dataset() #This is the eval InputDataset
        if len(self.train_dataset) > 500 and self.config.cache_images == "gpu": #we do not go in here
            CONSOLE.print(
                "Train dataset has over 500 images, overriding cache_images to cpu",
                style="bold yellow",
            )
            self.config.cache_images = "cpu"
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device #Do we want to keep images, masks, ... on device?
        if self.config.masks_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("mask") #Yes if you go in here
        if self.config.images_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("image") #Yes if you go in here

        # Some logic to make sure we sample every camera in equal amounts
        self.train_unseen_cameras = [i for i in range(len(self.train_dataset))] #[0,...,len(self.train_dataset)-1]
        self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))] #[0,...,len(self.eval_dataset)-1]
        assert len(self.train_unseen_cameras) > 0, "No data found in dataset"
        super().__init__()
        data_already_calculated = not os.path.exists("cached_data")
        if data_already_calculated:
            self.cached_eval_first_round[0]
            self.cached_train_first_round[0]
            # Save dataset.cameras to file
            self.add_all_color_ids()
        self.cached_eval[0]
        self.cached_train[0]
        self.map_id_color = self.cached_train[0]["id2map"] # type: ignore
        self.all_ids = self.cached_train[0]["all_ids"] # type: ignore
        if not os.path.exists("cached_data"):
            self.save_cached_data("cached_data")
            with open("cached_data/cameras_train.pkl", "wb") as f:
                pickle.dump(self.train_dataset.cameras, f)
            with open("cached_data/cameras_eval.pkl", "wb") as f:
                pickle.dump(self.eval_dataset.cameras, f)

        cmap = matplotlib.cm.get_cmap('tab20', len(self.all_ids))
        np.random.seed(0)
        indices = np.arange(len(self.all_ids))
        np.random.shuffle(indices)
        shuffled_cmap = cmap(indices)

        idxs = np.arange(len(self.all_ids))
        colors = shuffled_cmap[idxs]
        print(type(colors))
        map_id_color = {}
        for i in range(len(self.all_ids)):
            map_id_color[i] = colors[i][0:3]        
        print("Hallo")

    def save_cached_data(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.exists(os.path.join(directory, "cached_train")):
            os.makedirs(os.path.join(directory, "cached_train"))
        if not os.path.exists(os.path.join(directory, "cached_eval")):
            os.makedirs(os.path.join(directory, "cached_eval"))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(len(self.cached_eval)):
                data = self.cached_eval[i]
                filename = os.path.join(directory, f"cached_eval/cached_eval{i}.pkl")
                futures.append(executor.submit(pickle.dump, data, open(filename, "wb")))
            # Wait for all the futures to complete
            concurrent.futures.wait(futures)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(len(self.cached_train)):
                data = self.cached_train[i]
                filename = os.path.join(directory, f"cached_train/cached_train{i}.pkl")
                futures.append(executor.submit(pickle.dump, data, open(filename, "wb")))
            # Wait for all the futures to complete
            concurrent.futures.wait(futures)

    @cached_property
    def cached_train_first_round(self) -> List[Dict[str, torch.Tensor]]:
        """Get the training images. Will load and undistort the images the
        first time this (cached) property is accessed."""
        return self._load_images("train", cache_images_device=self.config.cache_images) #fetches all training images, undistorts them and caches them

    @cached_property
    def cached_eval_first_round(self) -> List[Dict[str, torch.Tensor]]:
        """Get the eval images. Will load and undistort the images the
        first time this (cached) property is accessed."""
        return self._load_images("eval", cache_images_device=self.config.cache_images) #fetches all training images, undistorts them and caches them
    
    @cached_property
    def cached_train(self)-> List[Dict[str, torch.Tensor]]:
        if not os.path.exists("cached_data/cached_train"):
            cached_train_first_round = deepcopy(self.cached_train_first_round)
            map_oldid_newid = {}
            for i in range(len(self.all_ids)):
                map_oldid_newid[self.all_ids[i]] = i
            #Now we need to change the ids in the image_inst_segm
            mask_out = map_oldid_newid.get(0,0)
            map_oldid_newid = np.vectorize(map_oldid_newid.get)  # Vectorize the dictionary lookup

            def parallel_map_oldid_newid(inst_segm_mask):
                return map_oldid_newid(inst_segm_mask)

            def inst_segm_masks_generator():
                for item in cached_train_first_round:
                    yield item["image_inst_segm"].numpy()

            num_cores = os.cpu_count()

            with ThreadPoolExecutor(max_workers=num_cores) as executor:
                #inst_segm_masks = [cached_train_first_round[i]["image_inst_segm"].numpy() for i in range(len(cached_train_first_round))]
                inst_segm_masks = list(executor.map(parallel_map_oldid_newid, inst_segm_masks_generator())) # type: ignore


            for i in range(len(cached_train_first_round)):
                cached_train_first_round[i]["image_inst_segm"] = torch.from_numpy(inst_segm_masks[i])
                #cached_train_first_round[i]["mask_img"] = (torch.from_numpy(inst_segm_masks[i]) == mask_out).float()

            # Run garbage collector
            gc.collect()

            for i in range(len(cached_train_first_round)):
                cached_train_first_round[i]["all_ids"] = self.all_ids # type: ignore
                cached_train_first_round[i]["id2map"] = self.map_id_color # type: ignore
                cached_train_first_round[i]["mask_out"] = mask_out

###############################################################################
#            num_files = len(os.listdir("cached_data_merged/cached_train"))
#            cached_train = []
#            with concurrent.futures.ThreadPoolExecutor() as executor:
#                futures = []
#                for i in range(0, num_files, 1):
#                    filename = f"cached_data_merged/cached_train/cached_train{i}.pkl"
#                    futures.append(executor.submit(pickle.load, open(filename, "rb")))
#                for future in concurrent.futures.as_completed(futures):
#                    data = future.result()
#                    cached_train.append(data)              
#            cached_train.sort(key=lambda x: x["image_idx"])

#            for i in range(len(cached_train)):
#                cached_train_first_round[i]["image_inst_segm"] = cached_train[i]["image_inst_segm"]       
###############################################################################            
            return cached_train_first_round
        else:
            cached_train = []
            num_files = len(os.listdir("cached_data/cached_train"))
            cached_train = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for i in range(0, num_files, 1):
                    filename = f"cached_data/cached_train/cached_train{i}.pkl"
                    futures.append(executor.submit(pickle.load, open(filename, "rb")))
                for future in concurrent.futures.as_completed(futures):
                    data = future.result()
                    cached_train.append(data)
            with open("cached_data/cameras_train.pkl", "rb") as f:
                self.train_dataset.cameras = pickle.load(f)
                self.train_dataset.cameras = self.train_dataset.cameras[::1]                    
            cached_train.sort(key=lambda x: x["image_idx"])
            self.train_unseen_cameras = [i for i in range(len(cached_train))]
            self.train_dataset = InputDatasetSkipped(1, self.train_dataset) # type: ignore
            #SimpleViewSelection(dataset=cached_train, cameras=self.train_dataset.cameras, Rmin=10.0, Tmin=1.0).select_views()
            return cached_train

    @cached_property
    def cached_eval(self)-> List[Dict[str, torch.Tensor]]:
        if not os.path.exists("cached_data/cached_eval"):
            cached_eval_first_round = deepcopy(self.cached_eval_first_round)
            map_oldid_newid = {}
            for i in range(len(self.all_ids)):
                map_oldid_newid[self.all_ids[i]] = i
            #Now we need to change the ids in the image_inst_segm
            #Now we need to change the ids in the image_inst_segm
            mask_out = map_oldid_newid.get(0, 0)
            map_oldid_newid = np.vectorize(map_oldid_newid.get)  # Vectorize the dictionary lookup

            def parallel_map_oldid_newid(inst_segm_mask):
                return map_oldid_newid(inst_segm_mask)

            
            with ThreadPoolExecutor() as executor:
                inst_segm_masks = [cached_eval_first_round[i]["image_inst_segm"].numpy() for i in range(len(cached_eval_first_round))]
                inst_segm_masks = list(executor.map(parallel_map_oldid_newid, inst_segm_masks))

            for i in range(len(cached_eval_first_round)):
                cached_eval_first_round[i]["image_inst_segm"] = torch.from_numpy(inst_segm_masks[i])
                #cached_eval_first_round[i]["mask_img"] = (torch.from_numpy(inst_segm_masks[i]) == mask_out).float()

            for i in range(len(cached_eval_first_round)):
                cached_eval_first_round[i]["all_ids"] = self.all_ids # type: ignore
                cached_eval_first_round[i]["id2map"] = self.map_id_color # type: ignore
                cached_eval_first_round[i]["mask_out"] = mask_out
            return cached_eval_first_round
        else:
            num_files = len(os.listdir("cached_data/cached_eval"))
            cached_eval = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for i in range(0, num_files, 1):
                    filename = f"cached_data/cached_eval/cached_eval{i}.pkl"
                    futures.append(executor.submit(pickle.load, open(filename, "rb")))
                for future in concurrent.futures.as_completed(futures):
                    data = future.result()
                    cached_eval.append(data)
            cached_eval.sort(key=lambda x: x["image_idx"])
            self.eval_dataset.cameras = self.eval_dataset.cameras[::1]
            self.eval_unseen_cameras = [i for i in range(len(cached_eval))]
            self.eval_dataset = InputDatasetSkipped(1, self.eval_dataset) # type: ignore
            #with open("cached_data/cameras_eval.pkl", "rb") as f:
            #    self.eval_dataset.cameras = pickle.load(f)
            return cached_eval

    def _load_images(
        self, split: Literal["train", "eval"], cache_images_device: Literal["cpu", "gpu"]
    ) -> List[Dict[str, torch.Tensor]]:
        undistorted_images: List[Dict[str, torch.Tensor]] = []

        # Which dataset?
        if split == "train":
            dataset = self.train_dataset #which dataset to use...
        elif split == "eval":
            dataset = self.eval_dataset #which dataset to use...
        else:
            assert_never(split)

        

        def undistort_idx(idx: int) -> Dict[str, torch.Tensor]: #undistort data at index idx
            data = dataset.get_data(idx, image_type=self.config.cache_images_type) #data contains image, mask and some other data
            camera = dataset.cameras[idx].reshape(()) #get camera
            assert data["image"].shape[1] == camera.width.item() and data["image"].shape[0] == camera.height.item(), (
                f'The size of image ({data["image"].shape[1]}, {data["image"].shape[0]}) loaded '
                f'does not match the camera parameters ({camera.width.item(), camera.height.item()})'
            )
            if camera.distortion_params is None:
                return data #if we deal with undistorted data, then we can just return the data as it is
            K = camera.get_intrinsics_matrices().numpy() #intrinsic parameters of camera
            distortion_params = camera.distortion_params.numpy() #the distortion of the camera
            image = data["image"].numpy() #just get the image

            K, image, mask, image_inst_segm = _undistort_image(camera, distortion_params, data, image, K, self.undistort_inst_seg_masks) #using camera, distortion_params, data, image, intrinsic parameters: undistort image and mask
            data["image"] = torch.from_numpy(image) #overwrite distorted image with undistorted image
            data["image_inst_segm"] = torch.from_numpy(image_inst_segm)
            if mask is not None:
                data["mask"] = mask #overwrite distorted mask with undistorted mask

            #change camera to match the undistorted images
            dataset.cameras.fx[idx] = float(K[0, 0])
            dataset.cameras.fy[idx] = float(K[1, 1])
            dataset.cameras.cx[idx] = float(K[0, 2])
            dataset.cameras.cy[idx] = float(K[1, 2])
            dataset.cameras.width[idx] = image.shape[1]
            dataset.cameras.height[idx] = image.shape[0]
            return data

        CONSOLE.log(f"Caching / undistorting {split} images")
        #undistort images with multiprocessing
        with ThreadPoolExecutor(max_workers=2) as executor:
            undistorted_images = list(
                track(
                    executor.map(
                        undistort_idx,
                        range(len(dataset)),
                    ),
                    description=f"Caching / undistorting {split} images",
                    transient=True,
                    total=len(dataset),
                )
            )

        # Move to device.
        if cache_images_device == "gpu":
            for cache in undistorted_images:
                cache["image"] = cache["image"].to(self.device)
                cache["image_inst_segm"] = cache["image_inst_segm"].to(self.device)
                if "mask" in cache:
                    cache["mask"] = cache["mask"].to(self.device)
        elif cache_images_device == "cpu":
            for cache in undistorted_images:
                cache["image"] = cache["image"].pin_memory()
                if "mask" in cache:
                    cache["mask"] = cache["mask"].pin_memory()
        else:
            assert_never(cache_images_device)


        return undistorted_images
    
    def add_all_color_ids(self):
        all_ids = set()
        eval_train = [self.cached_eval_first_round, self.cached_train_first_round]
        for imgs in eval_train:
            for i in range(len(imgs)): # type: ignore
                inst_segm_mask = imgs[i]["image_inst_segm"].numpy()
                object_ids_in_mask = np.unique(inst_segm_mask)
                object_ids_in_mask = object_ids_in_mask.tolist()
                all_ids.update(object_ids_in_mask)
        all_ids = list(all_ids)
        all_ids = sorted(all_ids)

        cmap = matplotlib.cm.get_cmap('tab20', len(all_ids))
        np.random.seed(2)
        indices = np.arange(len(all_ids))
        np.random.shuffle(indices)
        shuffled_cmap = cmap(indices)

        idxs = np.arange(len(all_ids))
        colors = shuffled_cmap[idxs]
        print(type(colors))
        map_id_color = {}
        for i in range(len(all_ids)):
            map_id_color[i] = colors[i][0:3]

        self.map_id_color = map_id_color
        self.all_ids = all_ids

    def create_train_dataset(self) -> TSegmentDataset:
        """Sets up the data loaders for training"""
        return self.dataset_type( #This is an InputDataset
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> TSegmentDataset:
        """Sets up the data loaders for evaluation"""
        return self.dataset_type( #This is an InputDataset
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    @cached_property
    def dataset_type(self) -> Type[TSegmentDataset]: #for splatfacto this returns the type InputDataset
        """Returns the dataset type passed as the generic argument"""
        default: Type[TSegmentDataset] = cast(TSegmentDataset, TSegmentDataset.__default__)  # type: ignore
        orig_class: Type[FullImageInstanceSegmentationDatamanager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is FullImageInstanceSegmentationDatamanager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is FullImageInstanceSegmentationDatamanager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is FullImageInstanceSegmentationDatamanager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TSegmentDataset], value)
        return default

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def setup_train(self):
        """Sets up the data loaders for training"""

    def setup_eval(self):
        """Sets up the data loader for evaluation"""

    @property
    def fixed_indices_eval_dataloader(self) -> List[Tuple[Cameras, Dict]]:
        """
        Pretends to be the dataloader for evaluation, it returns a list of (camera, data) tuples
        """
        #image_indices = [i for i in range(len(self.eval_dataset))]
        image_indices = [i for i in range(len(self.cached_eval))]
        data = deepcopy(self.cached_eval)
        _cameras = deepcopy(self.eval_dataset.cameras).to(self.device)
        cameras = []
        for i in image_indices:
            data[i]["image"] = data[i]["image"].to(self.device)
            data[i]["image_inst_segm"] = data[i]["image_inst_segm"].to(self.device)
            cameras.append(_cameras[i : i + 1])
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        return list(zip(cameras, data))

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}

    def get_train_rays_per_batch(self):
        # TODO: fix this to be the resolution of the last image rendered
        return 800 * 800

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        image_idx = self.train_unseen_cameras.pop(random.randint(0, len(self.train_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            #self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]
            self.train_unseen_cameras = [i for i in range(len(self.cached_train))]

        data = deepcopy(self.cached_train[image_idx])

        data["image"] = data["image"].to(self.device)
        data["image_inst_segm"] = data["image_inst_segm"].to(self.device)
        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle"""
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            #self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
            self.eval_unseen_cameras = [i for i in range(len(self.cached_eval))]
        data = deepcopy(self.cached_eval[image_idx])
        data["image"] = data["image"].to(self.device)
        data["image_inst_segm"] = data["image_inst_segm"].to(self.device)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        return camera, data

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle

        TODO: Make sure this logic is consistent with the vanilladatamanager"""
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            #self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
            self.eval_unseen_cameras = [i for i in range(len(self.cached_eval))]
        data = deepcopy(self.cached_eval[image_idx])
        data["image"] = data["image"].to(self.device)
        data["image_inst_segm"] = data["image_inst_segm"].to(self.device)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        return camera, data



"""
This method gets in 
-camera: The extrinsic and intrinsic parameters of the camera represented with the class Cameras
-distortion_params: How the fisheye camera distorts the image
-data: still do not know what this is
-image: probably the image which is supposed to be undistorted
-K: still do not know what this is
"""
def _undistort_image(
    camera: Cameras, distortion_params: np.ndarray, data: dict, image: np.ndarray, K: np.ndarray, undistort_inst_seg_masks: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[torch.Tensor], np.ndarray]:
    mask = None
    if camera.camera_type.item() == CameraType.PERSPECTIVE.value:
        raise NotImplementedError("CameraType.PERSPECTIVE.value has not been implemented yet")
    elif camera.camera_type.item() == CameraType.FISHEYE.value:
        K[0, 2] = K[0, 2] - 0.5
        K[1, 2] = K[1, 2] - 0.5
        distortion_params = np.array(
            [distortion_params[0], distortion_params[1], distortion_params[2], distortion_params[3]]
        )
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, distortion_params, (image.shape[1], image.shape[0]), np.eye(3), balance=0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, distortion_params, np.eye(3), newK, (image.shape[1], image.shape[0]), cv2.CV_32FC1 # type: ignore
        )
        # and then remap:
        image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
        if "mask" in data:
            mask = data["mask"].numpy()
            mask = mask.astype(np.uint8) * 255
            mask = cv2.fisheye.undistortImage(mask, K, distortion_params, None, newK)
            mask = torch.from_numpy(mask).bool()
            if len(mask.shape) == 2:
                mask = mask[:, :, None]
        image_inst_segm = data["image_inst_segm"].numpy()

        if undistort_inst_seg_masks:
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, distortion_params, np.eye(3), newK, 
                                                            (image_inst_segm.shape[1], image_inst_segm.shape[0]), cv2.CV_16SC2) # type: ignore

            # Undistort the segmentation mask using nearest-neighbor interpolation and border constant mode
            image_inst_segm = cv2.remap(image_inst_segm, map1, map2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            #image_inst_segm  = cv2.fisheye.undistortImage(image_inst_segm, K, distortion_params, None, newK)
            #test = test[:,:, None]
            image_inst_segm = image_inst_segm[:,:, None]
        #image_inst_segm = data["image_inst_segm"].numpy()
        #image_inst_segm = image_inst_segm[:,:, None]



        
        newK[0, 2] = newK[0, 2] + 0.5
        newK[1, 2] = newK[1, 2] + 0.5
        K = newK
    elif camera.camera_type.item() == CameraType.FISHEYE624.value:
        raise NotImplementedError("CameraType.FISHEYE624.value has not been implemented yet")

    else:
        raise NotImplementedError("Only perspective and fisheye cameras are supported")

    return K, image, mask, image_inst_segm
