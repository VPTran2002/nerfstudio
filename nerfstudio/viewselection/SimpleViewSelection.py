from collections import defaultdict
from nerfstudio.viewselection.BaseViewSelection import BaseViewSelection
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
import torch
from torchtyping import TensorType as TorchTensor
from typing import Dict, Tuple, List
from nerfstudio.cameras.cameras import Cameras, CameraType
from copy import deepcopy
import concurrent.futures
from PIL import Image
import numpy as np
import concurrent.futures


from tqdm import tqdm
import random
from tqdm import tqdm

def zero_out_close_points(points: torch.Tensor, farthest_point_index: torch.Tensor, Rdmat: torch.Tensor, Tdmat:torch.Tensor, Rmin: float, Tmin: float, distances: torch.Tensor) -> torch.Tensor:
    """
    Zero out the points that are close to the farthest point

    Parameters:
    points (torch.Tensor): Input point cloud of shape (N, 3)
    farthest_point_index (int): Index of the farthest point
    Rdmat (torch.Tensor): Distance matrix for rotation
    Tdmat (torch.Tensor): Distance matrix for translation
    Rmin (float): Minimum rotation distance
    Tmin (float): Minimum translation distance

    Returns:
    torch.Tensor: Zeroed out points of shape (N, 3)
    """
    N = points.shape[0]
    device = points.device

    # Get the indices where Rdmat[farthest_point_index][j] is smaller than Rmin
    indices_R = torch.where(Rdmat[farthest_point_index] < Rmin)[0]
    mask_R = torch.zeros(N, dtype=torch.bool, device=device)
    mask_R[indices_R] = True

    # Get the indices where Rdmat[farthest_point_index][j] is smaller than Rmin
    indices_T = torch.where(Tdmat[farthest_point_index] < Tmin)[0]   
    mask_T = torch.zeros(N, dtype=torch.bool, device=device)
    mask_T[indices_T] = True
    # Intersect indices_R and indices_T
    mask = mask_R & mask_T
    distances[mask] = 0
    indices_zeroed = torch.nonzero(mask).flatten()
    return indices_zeroed


def farthest_point_sampling(points: torch.Tensor, Rdmat: torch.Tensor, Tdmat:torch.Tensor, Rmin: float, Tmin: float, new_idx2old_idx: Dict) -> List[torch.Tensor]:
    """
    Perform farthest point sampling on a point cloud.

    Parameters:
    points (torch.Tensor): Input point cloud of shape (N, 3)
    num_samples (int): Number of points to sample

    Returns:
    torch.Tensor: Sampled points of shape (num_samples, 3)
    """
    N = points.shape[0]
    device = points.device
    # Randomly select the first point
    farthest_point_index = torch.randint(0, N, (1,), device=device)

    # Initialize an array to store the distances from the sampled points to all points
    distances = torch.full((N,), float('inf'), device=device)

    torch.zeros(N, device=device, dtype=points.dtype)

    clusters = []

    while torch.sum(distances) > 0:
        # Calculate the distances from the last sampled point to all points
        dist = torch.norm(points - points[farthest_point_index], dim=1)
        # Update the minimum distances
        distances = torch.min(distances, dist)
        # Select the farthest point from the set of sampled points
        farthest_point_index = torch.argmax(distances)

        #zero out the points close to the farthest point
        indices_zeroed = zero_out_close_points(points, farthest_point_index, Rdmat, Tdmat, Rmin, Tmin, distances)
        old_idx_zeroed = [new_idx2old_idx[k.item()] for k in indices_zeroed]
        clusters.append(old_idx_zeroed)
    clusters.sort(key=lambda x: x[0], reverse=True)
    return clusters


class SimpleViewSelection(BaseViewSelection):

    def __init__(self, dataset: List, cameras: Cameras, Rmin: float, Tmin: float, **kwargs):
        super().__init__(dataset, cameras, **kwargs)
        self.Rmin = Rmin
        self.Tmin = Tmin


    def Rt_dists(self, poses: torch.Tensor, deg=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """ 
        Compute the distance between the rotation and translation components of a 4x4 transformation matrix
        using the principal axes and Euclidean distance, respectively.
        """
        Rs1 = poses[:, :3, 0]
        Rs1 = Rs1 / torch.linalg.norm(Rs1, axis=1, keepdims=True)
        Rdmat1 = torch.einsum('mi,ni->mn', Rs1, Rs1)
        Rdmat1 = torch.arccos(torch.clip(Rdmat1, -1.0, 1.0))           
        Rs2 = poses[:, :3, 1]
        Rs2 = Rs2 / torch.linalg.norm(Rs2, axis=1, keepdims=True)
        Rdmat2 = torch.einsum('mi,ni->mn', Rs2, Rs2)
        Rdmat2 = torch.arccos(torch.clip(Rdmat2, -1.0, 1.0))          
        Rs3 = poses[:, :3, 2] # principal axes
        Rs3 = Rs3 / torch.linalg.norm(Rs3, axis=1, keepdims=True)
        Rdmat3 = torch.einsum('mi,ni->mn', Rs3, Rs3)
        Rdmat3 = torch.arccos(torch.clip(Rdmat3, -1.0, 1.0))        
        ts = poses[:, :3,  3]
        Tdmat = torch.linalg.norm(ts[:, None] - ts[None], axis=-1)

        if deg: 
            Rdmat3 = torch.rad2deg(Rdmat3)
            Rdmat2 = torch.rad2deg(Rdmat2)
            Rdmat1 = torch.rad2deg(Rdmat1)

        Rdmat = torch.max(torch.max(Rdmat3, Rdmat2), Rdmat1)

        return Rdmat, Tdmat

    def cluster_by_camera(self, idcs: List[int]):
        # Implement your logic here to cluster by camera
        viewmats, new_idx2old_idx = self.transform_cameras(idcs)
        Ts = viewmats[:, :3, 3]
        R_dist, T_dist = self.Rt_dists(viewmats)
        #Now, do farthest view sampling
        return farthest_point_sampling(Ts, R_dist, T_dist, self.Rmin, self.Tmin, new_idx2old_idx= new_idx2old_idx)

    def select_views_each_id(self, clusters):
        def default_value():
            return 0.0
        weights = defaultdict(default_value)
        for cluster in clusters:
            if cluster:
                random_index = random.randint(0, len(cluster) - 1)
                weights[cluster[random_index]] = 1.0
        return weights

    def select_views(self):
        # Implement your logic here to select views
        self.ids_in_img()
        instanceID2idxs = [set() for i in range(len(self.all_ids))]
        for i in range(len(self.dataset)):
            for j in self.dataset[i]["ids"]:
                instanceID2idxs[j].add(i)
        for i in range(len(instanceID2idxs)):
            instanceID2idxs[i] = list(instanceID2idxs[i]) # type: ignore
        instanceID2Clusters = []
        for i in range(len(instanceID2idxs)):
            instanceID2Clusters.append(self.cluster_by_camera(instanceID2idxs[i])) # type: ignore

        id2weights = []
        for i in range(len(instanceID2Clusters)):
            id2weights.append(self.select_views_each_id(instanceID2Clusters[i]))

        #Now calculate masks
        self.calculate_masks(id2weights)

        

        

        

