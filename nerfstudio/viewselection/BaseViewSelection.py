from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import torch
import concurrent.futures
import threading
from tqdm import tqdm

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras

class BaseViewSelection(ABC):

    def __init__(self, dataset, cameras, **kwargs):
        self.dataset = dataset
        self.cameras = cameras
        camera_optimizer_config = CameraOptimizerConfig()
        self.camera_optimizer = camera_optimizer_config.setup(
            num_cameras=len(dataset), device="cpu"
        )
        self.all_ids = dataset[0]["all_ids"]

    def obtain_viewmat_from_camera(self, camera: Cameras) -> torch.Tensor:
        optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)[0, ...]
        R = optimized_camera_to_world[:3, :3]  # 3 x 3
        T = optimized_camera_to_world[:3, 3:4]  # 3 x 1

        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=torch.device("cpu"), dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv

        return viewmat

    def transform_cameras(self, idcs) -> Tuple[torch.Tensor, Dict]:
        camera0 = self.cameras[0:1]
        viewmat0 = self.obtain_viewmat_from_camera(camera0)
        viewmats = torch.zeros((len(idcs), 4, 4), device=self.cameras.device, dtype=viewmat0.dtype)
        new_idx2old_idx = {}
        j = 0
        for i in idcs:
            camera = self.cameras[i:i+1]
            viewmat = self.obtain_viewmat_from_camera(camera)
            viewmats[j,:,:] = viewmat
            new_idx2old_idx[j] = i
            j += 1
        return viewmats, new_idx2old_idx
    
    def calculate_masks(self, id2weights: List[Dict[int, float]]):
        def calculate_mask(i):
            inst_segm = self.dataset[i]["image_inst_segm"]
            ids = self.dataset[i]["ids"]
            mask = torch.zeros(inst_segm.shape, dtype=torch.float32)
            for id in ids:
                if id2weights[id][i] == 0.0:
                    continue
                mask += id2weights[id][i] * (inst_segm == id).float()
            self.dataset[i]["mask_img"] = mask
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            with tqdm(total=len(self.dataset), desc="Calculating masks") as pbar:
                futures = [executor.submit(calculate_mask, i) for i in range(len(self.dataset))]
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(1)    

    def ids_in_img(self):
        """
        Get the indices of the points in the image

        Parameters:
        img_idx (int): Index of the image

        Returns:
        torch.Tensor: Indices of the points in the image
        """
        instance_idx2image_idx = [set() for i in range(len(self.all_ids))]
        lock = threading.Lock()
        def process_data(i):
            seg_mask = self.dataset[i]["image_inst_segm"]
            unique_values = torch.unique(seg_mask)
            self.dataset[i]["ids"] = unique_values
            with lock:
                for id in unique_values:
                    instance_idx2image_idx[id].add(i)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            with tqdm(total=len(self.dataset), desc="Processing data") as pbar:
                futures = [executor.submit(process_data, i) for i in range(len(self.dataset))]
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(1)

        return instance_idx2image_idx

    """
    """    
    @abstractmethod
    def select_views(self):
        pass