import os
import pickle
import cv2
import tyro
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion
from nerfstudio.scripts.loader import get_trainer
import torch
import matplotlib.pyplot as plt
import concurrent.futures
import matplotlib.cm
import numpy as np
import h5py
import math


import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image


colors = [[128, 0, 0], [0, 128, 0], [128, 128, 0],
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
"""
colors = [[128, 0, 0], [0, 128, 0], [128, 128, 0],
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
"""
def save_boolean_array_to_png(boolean_array, file_path):
    # Convert boolean array to uint8
    uint8_array = (boolean_array * 255).astype(np.uint8)
    
    # Create an Image object from the array
    image = Image.fromarray(uint8_array)
    
    # Save the image
    image.save(file_path)


def calculate_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return intersection / union if union != 0 else 0

def mean_iou(gt, pred):
    gt_labels = np.unique(gt)
    pred_labels = np.unique(pred)
    H,W=gt.shape
    size=H*W
    # Remove the background label (assuming it's 0)
    gt_labels = gt_labels[gt_labels != 0]
    gt_labels = [label for label in gt_labels if np.sum(gt == label) > size*0.001]
    
    iou_matrix = np.zeros((len(gt_labels), len(pred_labels)))
    
    for i, gt_label in enumerate(gt_labels):
        gt_mask = (gt == gt_label)
        for j, pred_label in enumerate(pred_labels):
            pred_mask = (pred == pred_label)
            iou_matrix[i, j] = calculate_iou(gt_mask, pred_mask)
    
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # We use -iou_matrix to maximize the sum
    total_iou = iou_matrix[row_ind, col_ind].sum()
    mean_iou_value = total_iou / len(gt_labels)  # Normalize by the number of ground truth instances
    if math.isnan(mean_iou_value):
        mean_iou_value = 1.0
    return mean_iou_value, iou_matrix, gt_labels, pred_labels

#np.random.shuffle(colors)

def _mask_to_rgb_image(mask, map_id_color):

    H,W = mask.shape
    rgb_image = torch.zeros((H, W, 3), dtype=torch.uint8)
    #for object_id, color in map_id_color.items():
    # Randomly shuffle the colors
    for object_id, color in enumerate(colors):

        # Create a boolean mask where mask equals the current object_id
        object_mask = (mask == object_id)

        # Assign the corresponding color to the pixels where the object_mask is True
        for channel in range(3):
            rgb_image[:, :, channel][object_mask] = color[channel]

    return rgb_image

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    trainer = get_trainer(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__), # type: ignore
        ) # type: ignore
    )
    with open("cached_data_gt_0a184cf634/cameras_train.pkl", "rb") as f:
        cameras = pickle.load(f)

    #ground_truth_images
    #cmap = matplotlib.cm.get_cmap('tab20', 141)
    #np.random.seed(2)
    #indices = np.arange(141)
    #np.random.shuffle(indices)
    #shuffled_cmap = cmap(indices)

    #idxs = np.arange(141)
    #colors = shuffled_cmap[idxs]
    #print(type(colors))
    #map_id_color = {}
    #for i in range(141):
    #    map_id_color[i] = colors[i][0:3]
    
    trainer.pipeline.train()
    model = trainer.pipeline.model
    cameras = trainer.pipeline.datamanager.train_dataset.cameras

    directory = "cached_data_gt_0a184cf634/cached_train"
    num_files = len(os.listdir(directory))
    cached_train = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, num_files, 1):
            filename = directory+f"/cached_train{i}.pkl"
            futures.append(executor.submit(pickle.load, open(filename, "rb")))
        for future in concurrent.futures.as_completed(futures):
            data = future.result()
            cached_train.append(data)             
    cached_train.sort(key=lambda x: x["image_idx"])

    camera = cameras[0:1].to(torch.device("cuda:0"))
    output = model.get_outputs(camera)
    mask=output["instance_segmentation_mask"].numpy()
    directory_debug = "debug_GT_loading"
    IoU = 0
    with h5py.File("masks_imgs_dict.h5py", 'a') as your_file:
        for i in range(len(cameras)):
            camera = cameras[i:i+1].to(torch.device("cuda:0"))
            output = model.get_outputs(camera)
            rgb = output["rgb"].squeeze().to("cpu").detach().numpy()
            mask=torch.argmax(output["instance_segmentation_logits"], dim=2)
            gt = cached_train[i]["image_inst_segm"].squeeze()
            mean_iou_value, iou_matrix, _, _ = mean_iou(gt.to("cpu").numpy(), mask.to("cpu").numpy())
            
            mask_cpu = mask.cpu().numpy()

            your_file.create_group(str(i))
            your_file[str(i)].create_dataset("masks", data=gt.to("cpu").numpy()) # type: ignore
            your_file[str(i)].create_dataset("image", data=cached_train[i]["image"]) # type: ignore

            mask = _mask_to_rgb_image(mask, None).numpy()
            gt = _mask_to_rgb_image(gt, None).numpy()
            #gt = _mask_to_rgb_image(gt, None).numpy()
            #debug_image = np.concatenate((mask, gt), axis=1)
            IoU += mean_iou_value
            plt.imsave(directory_debug + '/'+ str(i) + "_iou_"+str(mean_iou_value)+".png", mask)
            plt.imsave(directory_debug + '/'+ str(i) + "_normal.png", rgb)
            plt.imsave(directory_debug + '/'+ str(i) + "_gt_instance.png", gt)
            plt.imsave(directory_debug + '/'+ str(i) + "_gt_rgb.png", cached_train[i]["image"].squeeze().to("cpu").detach().numpy())


        #plt.imsave(gt_file, gt.numpy())
    print(IoU/len(cameras))


    with open("cached_data_gt_0a184cf634/cameras_eval.pkl", "rb") as f:
        cameras = pickle.load(f)

    #ground_truth_images
    #cmap = matplotlib.cm.get_cmap('tab20', 141)
    #np.random.seed(2)
    #indices = np.arange(141)
    #np.random.shuffle(indices)
    #shuffled_cmap = cmap(indices)

    #idxs = np.arange(141)
    #colors = shuffled_cmap[idxs]
    #print(type(colors))
    #map_id_color = {}
    #for i in range(141):
    #    map_id_color[i] = colors[i][0:3]
    
    trainer.pipeline.train()
    model = trainer.pipeline.model
    cameras = trainer.pipeline.datamanager.eval_dataset.cameras

    directory = "cached_data_gt_0a184cf634/cached_eval"
    num_files = len(os.listdir(directory))
    cached_eval = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, num_files, 1):
            filename = directory+f"/cached_eval{i}.pkl"
            futures.append(executor.submit(pickle.load, open(filename, "rb")))
        for future in concurrent.futures.as_completed(futures):
            data = future.result()
            cached_eval.append(data)             
    cached_eval.sort(key=lambda x: x["image_idx"])

    camera = cameras[0:1].to(torch.device("cuda:0"))
    output = model.get_outputs(camera)
    mask=output["instance_segmentation_mask"].numpy()
    directory_debug = "debug_GT_eval_loading"
    IoU = 0
    for i in range(len(cameras)):
        camera = cameras[i:i+1].to(torch.device("cuda:0"))
        output = model.get_outputs(camera)
        rgb = output["rgb"].squeeze().to("cpu").detach().numpy()
        mask=torch.argmax(output["instance_segmentation_logits"], dim=2)
        gt = cached_eval[i]["image_inst_segm"].squeeze()
        mean_iou_value, iou_matrix, _, _ = mean_iou(gt.to("cpu").numpy(), mask.to("cpu").numpy())
        mask = _mask_to_rgb_image(mask, None).numpy()
        #gt = _mask_to_rgb_image(gt, None).numpy()
        #debug_image = np.concatenate((mask, gt), axis=1)
        print(i)
        print(mean_iou_value)
        IoU += mean_iou_value
        plt.imsave(directory_debug + '/'+ str(i) + "_iou_"+str(mean_iou_value)+".png", mask)
        plt.imsave(directory_debug + '/'+ str(i) + "_normal.png", rgb)

        #plt.imsave(gt_file, gt.numpy())
    print(IoU/len(cameras))

          


if __name__ == "__main__":
    entrypoint()