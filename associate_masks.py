import os
import pickle
from typing import Dict, Tuple
from matplotlib import pyplot as plt
import numpy as np
import cv2
from collections import Counter

import torch

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
import concurrent.futures
import matplotlib.cm

colors = [[0, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128], [128, 64, 128], [0, 192, 128], [128, 192, 128],
    [64, 64, 0], [192, 64, 0], [64, 192, 0], [192, 192, 0],
    [64, 64, 128], [192, 64, 128], [64, 192, 128], [192, 192, 128],
    [0, 0, 64], [128, 0, 64], [0, 128, 64], [128, 128, 64],
    [0, 0, 192], [128, 0, 192], [0, 128, 192], [128, 128, 192],
    [64, 0, 64], [192, 0, 64], [64, 128, 64], [192, 128, 64],
    [64, 0, 192], [192, 0, 192], [64, 128, 192], [192, 128, 192],
    [0, 64, 64], [128, 64, 64], [0, 192, 64], [128, 192, 64],
    [0, 64, 192], [128, 64, 192], [0, 192, 192], [128, 192, 192],
    [64, 64, 64], [192, 64, 64], [64, 192, 64], [192, 192, 64],
    [64, 64, 192], [192, 64, 192], [64, 192, 192], [192, 192, 192],
    [32, 0, 0], [160, 0, 0], [32, 128, 0], [160, 128, 0],
    [32, 0, 128], [160, 0, 128], [32, 128, 128], [160, 128, 128],
    [96, 0, 0], [224, 0, 0], [96, 128, 0], [224, 128, 0],
    [96, 0, 128], [224, 0, 128], [96, 128, 128], [224, 128, 128],
    [32, 64, 0], [160, 64, 0], [32, 192, 0], [160, 192, 0],
    [32, 64, 128], [160, 64, 128], [32, 192, 128], [160, 192, 128],
    [96, 64, 0], [224, 64, 0], [96, 192, 0], [224, 192, 0],
    [96, 64, 128], [224, 64, 128], [96, 192, 128], [224, 192, 128],
    [32, 0, 64], [160, 0, 64], [32, 128, 64], [160, 128, 64],
    [32, 0, 192], [160, 0, 192], [32, 128, 192], [160, 128, 192],
    [96, 0, 64], [224, 0, 64], [170, 170, 170], [224, 128, 64],
    [96, 0, 192], [224, 0, 192], [96, 128, 192], [224, 128, 192],
    [32, 64, 64], [160, 64, 64], [32, 192, 64], [160, 192, 64],
    [32, 64, 192], [160, 64, 192], [32, 192, 192], [160, 192, 192],
    [96, 64, 64], [224, 64, 64], [96, 192, 64], [224, 192, 64],
    [29, 195, 49], [224, 64, 192], [96, 192, 192], [224, 192, 192],
    [0, 32, 0], [128, 32, 0], [0, 160, 0], [128, 160, 0],
    [0, 32, 128], [128, 32, 128], [0, 160, 128], [128, 160, 128],
    [64, 32, 0], [192, 32, 0], [64, 160, 0], [192, 160, 0],
    [64, 32, 128], [192, 32, 128], [64, 160, 128], [192, 160, 128],
    [0, 96, 0], [128, 96, 0], [0, 224, 0], [128, 224, 0],
    [0, 96, 128], [128, 96, 128], [0, 224, 128], [128, 224, 128],
    [64, 96, 0], [192, 96, 0], [64, 224, 0], [54, 62, 167],
    [64, 96, 128], [95, 219, 255], [64, 224, 128], [192, 224, 128],
    [0, 32, 64], [128, 32, 64], [0, 160, 64], [128, 160, 64],
    [0, 32, 192], [128, 32, 192], [0, 160, 192], [128, 160, 192],
    [64, 32, 64], [140, 104, 47], [64, 160, 64], [192, 160, 64],
    [64, 32, 192], [192, 32, 192], [64, 160, 192], [192, 160, 192],
    [0, 96, 64], [128, 96, 64], [0, 224, 64], [128, 224, 64],
    [0, 96, 192], [128, 96, 192], [0, 224, 192]
]

def _downscale_segmentation_mask(mask):
    d = 4
    downscaled_mask = cv2.resize(mask, (mask.shape[1]//d, mask.shape[0]//d), interpolation=cv2.INTER_NEAREST)
    return downscaled_mask

def _mask_to_rgb_image(mask, map_id_color):
    H,W = mask.shape
    rgb_image = torch.zeros((H, W, 3), dtype=torch.uint8)
    #for object_id, color in map_id_color.items():
    for object_id, color in enumerate(colors):

        # Create a boolean mask where mask equals the current object_id
        object_mask = (mask == object_id)

        # Assign the corresponding color to the pixels where the object_mask is True
        for channel in range(3):
            rgb_image[:, :, channel][object_mask] = color[channel]

    return rgb_image

camera_optimizer_config = CameraOptimizerConfig()
camera_optimizer = camera_optimizer_config.setup(
    num_cameras=308, device="cpu"
)

def consolidate_instance_ids(all_instance_ids):
    consolidated_ids = []
    for i in range(len(all_instance_ids[0])):
        instance_votes = [instance_ids[i].item() for instance_ids in all_instance_ids if instance_ids[i] != -1]
        if instance_votes:
            most_common_instance = Counter(instance_votes).most_common(1)[0][0]
            consolidated_ids.append(most_common_instance)
        else:
            consolidated_ids.append(-1)  # No valid instance ID
    return consolidated_ids

def project_points(points_3d, camera_intrinsic, camera_extrinsic):
    # Convert 3D points to homogeneous coordinates
    points_3d_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    # Project points to 2D
    points_camera = camera_extrinsic @ points_3d_hom.T  # [4, N]
    
    # Discard the homogeneous coordinate (4th row)
    points_camera = points_camera[:3, :]  # [3, N]
    
    # Project points to 2D image plane
    points_2d_hom = camera_intrinsic @ points_camera  # [3, N]
    points_2d = points_2d_hom[:2, :] / points_2d_hom[2, :]
    return points_2d.T

def associate_instances(points_2d, instance_mask):
    instance_ids = []
    for point in points_2d:
        x, y = int(point[0]), int(point[1])
        if 0 <= y < instance_mask.shape[0] and 0 <= x < instance_mask.shape[1] and instance_mask[y, x] != 0:
            instance_ids.append(instance_mask[y, x])
        else:
            instance_ids.append(-1)  # Outside image boundaries
    return instance_ids

def obtain_viewmat_and_intrinsic_from_camera(camera: Cameras) -> Tuple[torch.Tensor,torch.Tensor]:
    optimized_camera_to_world = camera_optimizer.apply_to_camera(camera)[0, ...]
    camera.rescale_output_resolution(1 / 4)
    R = optimized_camera_to_world[:3, :3]  # 3 x 3
    T = optimized_camera_to_world[:3, 3:4]  # 3 x 1

    # flip the z and y axes to align with gsplat conventions
    R_edit = torch.diag(torch.tensor([1, -1, -1], device=torch.device("cpu"), dtype=R.dtype))
    R = R @ R_edit
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.T
    T_inv = -R_inv @ T
    R_inv=R
    T_inv=T
    viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
    viewmat[:3, :3] = R_inv
    viewmat[:3, 3:4] = T_inv

    K = camera.get_intrinsics_matrices()

    return viewmat, K

def transform_cameras(cameras,idcs) -> Tuple[torch.Tensor, torch.Tensor]:
    camera0 = cameras[0:1]
    viewmat0, K = obtain_viewmat_and_intrinsic_from_camera(camera0)
    viewmats = torch.zeros((len(idcs), 4, 4), device=cameras.device, dtype=viewmat0.dtype)
    Ks = torch.zeros((len(idcs), 3, 3), device=cameras.device, dtype=viewmat0.dtype)
    j = 0
    for i in idcs:
        camera = cameras[i:i+1]
        viewmat, K = obtain_viewmat_and_intrinsic_from_camera(camera)
        viewmats[j,:,:] = viewmat
        Ks[j,:,:] = K
        j += 1
    return viewmats, Ks

def overwrite_instance_ids(instance_mask, points_2d, instance_ids):
    """
    Overwrites instance IDs in the segmentation mask based on majority label from 3D points.

    Args:
        instance_mask (np.ndarray): 2D array of instance IDs.
        points_2d (np.ndarray): Nx2 array of 2D projected points.
        instance_ids (list of int): List of instance IDs corresponding to the 2D points.

    Returns:
        np.ndarray: Modified instance mask with updated instance IDs.
    """
    instance_id_map = {}
    for point, inst_id in zip(points_2d, instance_ids):
        x, y = int(point[0]), int(point[1])
        if 0 <= y < instance_mask.shape[0] and 0 <= x < instance_mask.shape[1]:
            if inst_id != -1 and instance_mask[y, x].item() != 0:
                if instance_mask[y, x].item() not in instance_id_map:
                    instance_id_map[instance_mask[y, x].item()] = []
                instance_id_map[instance_mask[y, x].item()].append(inst_id)
    
    for inst_id, votes in instance_id_map.items():
        if votes:
            majority_label = Counter(votes).most_common(1)[0][0]
            instance_mask[instance_mask == inst_id] = majority_label
    
    return instance_mask


# Example data
checkpoint_file = 'means_colors_079a326597.ckpt'
means_colors = torch.load(checkpoint_file)
points_3d = means_colors['means'].to(torch.device('cpu')).numpy()

with open("cached_data/cameras_train.pkl", "rb") as f:
    cameras = pickle.load(f)
idcs = [i for i in range(len(cameras))]
camera_poses, camera_intrinsic = transform_cameras(cameras, idcs=idcs)
camera_poses, camera_intrinsic = camera_poses.to(torch.device('cpu')).numpy(), camera_intrinsic.to(torch.device('cpu')).numpy()

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
cached_train.sort(key=lambda x: x["image_idx"])
instance_masks = [cached_train[i]["image_inst_segm"].numpy() for i in range(len(cached_train))] 
#points_3d = np.array([...])  # Nx3 array of 3D points
#camera_intrinsic = np.array([...])  # 3x3 intrinsic matrix
#camera_extrinsic = np.array([...])  # 4x4 extrinsic matrix
#instance_masks = [np.array([...]), ...]  # List of instance segmentation masks

all_instance_ids = []

i = 0
for camera_pose, instance_mask in zip(camera_poses, instance_masks):
    instance_mask = _downscale_segmentation_mask(instance_mask)
    points_2d = project_points(points_3d, camera_intrinsic[i], camera_pose)
    instance_ids = associate_instances(points_2d, instance_mask)
    all_instance_ids.append(instance_ids)
    i += 1

consolidated_ids = consolidate_instance_ids(all_instance_ids)

"""
cmap = matplotlib.cm.get_cmap('tab20', 141)
np.random.seed(2)
indices = np.arange(141)
np.random.shuffle(indices)
shuffled_cmap = cmap(indices)

idxs = np.arange(141)
colors = shuffled_cmap[idxs]
print(type(colors))
map_id_color = {}
for i in range(141):
    map_id_color[i] = colors[i][0:3]
"""

#project 
for i, (camera_pose, instance_mask) in enumerate(zip(camera_poses, instance_masks)):
    instance_mask = _downscale_segmentation_mask(instance_mask)
    points_2d = project_points(points_3d, camera_intrinsic[i], camera_pose)
    rgb_before = _mask_to_rgb_image(instance_mask.squeeze(), None)
    instance_mask = overwrite_instance_ids(instance_mask, points_2d, consolidated_ids)
    rgb = _mask_to_rgb_image(instance_mask.squeeze(), None)
    mask_file = f"/usr/stud/tranv/storage/tranv/Research/OOD/BE5DW/nerfstudio/Test/mask_3D{i}.png"
    before = f"/usr/stud/tranv/storage/tranv/Research/OOD/BE5DW/nerfstudio/Test2/mask_3D{i}.png"
    plt.imsave(mask_file, rgb.numpy())
    plt.imsave(before, rgb_before.numpy())
    print(f"Updated instance mask for view {i}:\n", instance_mask)
