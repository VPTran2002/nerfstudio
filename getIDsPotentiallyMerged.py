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

def _old_mask_to_new_mask(mask, map_id_color):
    H,W = mask.shape
    new_mask = torch.zeros((H, W), dtype=torch.uint8)
    #for object_id, color in map_id_color.items():
    # Randomly shuffle the colors
    for old_value, new_value in map_id_color.items():

        # Create a boolean mask where mask equals the current object_id
        object_mask = (mask == old_value)

        # Assign the corresponding color to the pixels where the object_mask is True
        for channel in range(3):
            new_mask[object_mask] = new_value

    return new_mask

def find_overlapping_instances(M1, M2, mappper):
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

def create_consistent_images(M1, M2, mappper):
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

        


"""
def _mask_to_rgb_image(mask, map_id_color):
    H,W = mask.shape
    rgb_image = torch.zeros((H, W, 3), dtype=torch.uint8)
    for object_id, color in map_id_color.items():
        # Create a boolean mask where mask equals the current object_id
        object_mask = (mask == object_id)

        # Assign the corresponding color to the pixels where the object_mask is True
        for channel in range(3):
            rgb_image[:, :, channel][object_mask] = color[channel]*255

    return rgb_image
"""

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
    with open("cached_data/cameras_train.pkl", "rb") as f:
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
    

    model = trainer.pipeline.model

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

    camera = cameras[0:1].to(torch.device("cuda:0"))
    output = model.get_outputs(camera)
    mask=output["instance_segmentation_mask"].numpy()
    #height, width = mask.shape[:2]
    #fps = 5.0  # Frames per second
    #out_0 = cv2.VideoWriter('consistent_training_signal.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) # type: ignore
    #out_1 = cv2.VideoWriter('consistent_training_signal_before.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) # type: ignore
    #out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) # type: ignore

    mapper = {}

    for i in range(len(cameras)):
        mask_consistent = cached_train[i]["image_inst_segm"].squeeze()
        camera = cameras[i:i+1].to(torch.device("cuda:0"))
        output = model.get_outputs(camera)
        mask=torch.argmax(output["instance_segmentation_logits"], dim=2)
        find_overlapping_instances(mask_consistent.to(torch.device("cpu")).numpy(), mask.to(torch.device("cpu")).numpy(), mapper)

    for i in range(len(cameras)):
        mask_consistent = cached_train[i]["image_inst_segm"].squeeze()
        camera = cameras[i:i+1].to(torch.device("cuda:0"))
        output = model.get_outputs(camera)
        mask=torch.argmax(output["instance_segmentation_logits"], dim=2)
        new_mask = create_consistent_images(mask_consistent.to(torch.device("cpu")).numpy(), mask.to(torch.device("cpu")).numpy(), mapper)
        cached_train[i]["image_inst_segm"] = torch.from_numpy(new_mask).squeeze()
        #np.save(f"consistent_masks/consistent_mask_{i}.npy", new_mask)
        #out_0.write(_mask_to_rgb_image(torch.from_numpy(new_mask), None).numpy())
        #out_1.write(_mask_to_rgb_image(mask_consistent, None).numpy())

    directory = "cached_data_merged"
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(len(cached_train)):
            data = cached_train[i]
            filename = os.path.join(directory, f"cached_train/cached_train{i}.pkl")
            futures.append(executor.submit(pickle.dump, data, open(filename, "wb")))
        # Wait for all the futures to complete
        concurrent.futures.wait(futures)

"""
    out_1.release()
    out_0.release()
    print("Video saved as consistent_training_signal.mp4")

    for i in range(len(cameras)):
        camera = cameras[i:i+1].to(torch.device("cuda:0"))
        output = model.get_outputs(camera)
        mask=torch.argmax(output["instance_segmentation_logits"], dim=2)
        #mask = _old_mask_to_new_mask(mask, map_old2new_id)
        mask = _mask_to_rgb_image(mask, None).numpy()
        #gt=_mask_to_rgb_image(cached_train[i]["image_inst_segm"].squeeze(), map_id_color)
        # Save mask to file
        #gt_file = "gt.png"
        #mask_file = "mask.png"
        out.write(mask)

        # Release the VideoWriter object
    out.release()
    print("Video saved as output.mp4")
        #plt.imsave(mask_file, mask)
        #plt.imsave(gt_file, gt.numpy())

          

"""
if __name__ == "__main__":
    entrypoint()
