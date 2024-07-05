import h5py
import matplotlib.cm
import matplotlib.image
import torch
import numpy as np

path_to_h5 = "/usr/stud/tranv/storage/tranv/Research/OOD/BE5DW/scannetppData/NouriSendsLabels/VeryFuckingBAD079a326598.h5"#groundingSAM_instance_masks_5000th_5C_079a326597.h5"
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

def sort_indices_with_duplicates(a1, a2):
    # Step 1: Create a mapping from elements of a1 to their first occurrence index
    order_dict = {}
    for idx, val in enumerate(a1):
        if val not in order_dict:
            order_dict[val] = idx
    
    # Step 2: Create a list of tuples (value, index) for a2
    indexed_a2 = [(val, idx) for idx, val in enumerate(a2)]
    
    # Step 3: Sort indexed_a2 based on the order in a1
    sorted_indexed_a2 = sorted(indexed_a2, key=lambda x: order_dict[x[0]])
    
    # Step 4: Extract the sorted indices
    sorted_indices = [idx for val, idx in sorted_indexed_a2]
    
    return sorted_indices

cmap = matplotlib.cm.get_cmap('tab20', 141)
np.random.seed(2)
indices = np.arange(141)
np.random.shuffle(indices)
shuffled_cmap = cmap(indices)

idxs = np.arange(141)
colors = shuffled_cmap[idxs]
#print(type(colors))
map_id_color = {}
for i in range(141):
    map_id_color[i] = colors[i][0:3]
# Read the h5 file
with h5py.File(path_to_h5, 'r') as file:
    keys = list(file.keys())
    # Convert h5py dataset to numpy array
    j = 0
    for key in keys:
        print(j)
        j+=1
        debug = file[key]
        print(list(debug.keys()))
        order = file[key]["order"][:]
        segmentation_array = file[key]["segmentation"][:]
        labels = file[key]["labels"][:]
        sorted_indices = sort_indices_with_duplicates(order, labels)
        segmentation_mask = np.zeros((segmentation_array.shape[1], segmentation_array.shape[2]), dtype=np.uint8)
        for idx in sorted_indices:
            segmentation_mask[segmentation_array[idx].squeeze()] = labels[idx] + 1
        #img_colored = _mask_to_rgb_image(segmentation_mask, map_id_color).numpy()
        # Save the colored mask as a PNG file
        #output_path = "test.png"
        #matplotlib.image.imsave(output_path, img_colored)
        output_path = "/usr/stud/tranv/storage/tranv/Research/OOD/BE5DW/scannetppData/DownloadedScenes/data_segmentation/079a326597_fucking_bad/instance_segmentation/resized_images/" + key[:-4] + ".npy"
        np.save(output_path, segmentation_mask)

        
    #print(type(segmentation_array))