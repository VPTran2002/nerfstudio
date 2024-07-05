import numpy as np
import torch
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import pickle

def save_point_cloud_to_ply(points, filename):
    # Ensure points is a numpy array of shape (N, 3)
    if isinstance(points, np.ndarray) and points.shape[1] == 3:
        points = [(point[0], point[1], point[2]) for point in points]

        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        ply_element = PlyElement.describe(vertex, 'vertex')
        PlyData([ply_element], text=True).write(filename)
    else:
        raise ValueError("Input tensor must be a numpy array with shape (N, 3)")



#with open("cached_data/cameras_eval.pkl", "rb") as f:
#    q = pickle.load(f)  
# Get the CUDA device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load the checkpoint file
checkpoint_file = '/usr/stud/tranv/storage/tranv/Research/OOD/BE5DW/nerfstudio/means_colors_scores_079a326597_fucking_bad.ckpt'
means_colors = torch.load(checkpoint_file)
print("Hallo")
#means = means_colors['means'].to(torch.device('cpu')).numpy()

# Example usage
#save_point_cloud_to_ply(means, '/usr/stud/tranv/storage/tranv/Research/OOD/BE5DW/nerfstudio/means.ply')


