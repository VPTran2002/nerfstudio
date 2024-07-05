import h5py
import numpy as np
import torch
import matplotlib

import matplotlib.cm
import matplotlib.image as mpimg

def _mask_to_rgb_image(mask, map_id_color):
    mask = mask.squeeze(0)
    H,W = mask.shape
    rgb_image = torch.zeros((H, W, 3), dtype=torch.uint8)
    for object_id, color in map_id_color.items():
        # Create a boolean mask where mask equals the current object_id
        object_mask = (mask == object_id)

        # Assign the corresponding color to the pixels where the object_mask is True
        for channel in range(3):
            rgb_image[:, :, channel][object_mask] = color[channel]*255

    return rgb_image

def print_h5_structure(name, obj):
    """
    Recursively prints the structure of an HDF5 file.
    
    Parameters:
    - name: The name of the current object.
    - obj: The current object (Group or Dataset).
    """
    indent = ' ' * (len(name.split('/')) - 1) * 2
    if isinstance(obj, h5py.Group):
        print(f"{indent}Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")

def transform_masks_to_images(masks, mask_labels):
    result_image = torch.zeros((1, masks.shape[2], masks.shape[3])) - 2
    for i in range(masks.shape[0]):
        overlap = (masks[i] & (result_image > -2)).any()
        if overlap:
            print("Overlap detected")
        result_image = torch.where(masks[i], mask_labels[i], result_image)
    return result_image


path_to_h5 = "/usr/stud/tranv/storage/tranv/Research/OOD/BE5DW/scannetppData/NouriSendsLabels/labeled_masks_gSAM_min_normal.h5"
save_dir = "/usr/stud/tranv/storage/tranv/Research/OOD/BE5DW/scannetppData/DownloadedScenes/data_segmentation/079a326597/resized_images"

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
# Open the H5 file
with h5py.File(path_to_h5, "r") as file:
    # Print the structure of the file
    #file.visititems(print_h5_structure)
    for key in file.keys():
        print(key)
        masks = torch.from_numpy(np.array(file[key + '/masks']))
        mask_labels_tensor = torch.from_numpy(np.array(file[key + '/masks_label']))
        mask_transformed = transform_masks_to_images(masks, mask_labels_tensor)
        mask_transformed = mask_transformed.squeeze(0)
        np.save(save_dir + '/' + key[:-4] + ".npy", mask_transformed.numpy())
        #rgb = _mask_to_rgb_image(mask_transformed, map_id_color) NOTE: This is just for visualization
        #np_rgb = rgb.numpy()

        
# Remember to close the file after you're done
