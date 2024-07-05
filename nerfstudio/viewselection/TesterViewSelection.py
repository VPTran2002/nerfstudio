from typing import Any, Dict, List, Tuple
from nerfstudio.viewselection.BaseViewSelection import BaseViewSelection
from nerfstudio.viewselection.SimpleViewSelection import SimpleViewSelection
import os
import concurrent.futures
import pickle
from nerfstudio.cameras.cameras import Cameras, CameraType
import torch

def load_test_data() -> List[Dict[str, Any]]:
    cached_train = []
    num_files = len(os.listdir("cached_data/cached_train"))
    cached_train = []
    cameras = None
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(num_files):
            filename = f"cached_data/cached_train/cached_train{i}.pkl"
            futures.append(executor.submit(pickle.load, open(filename, "rb")))
        for future in concurrent.futures.as_completed(futures):
            data = future.result()
            cached_train.append(data)                 
    cached_train.sort(key=lambda x: x["image_idx"])
    return cached_train

def load_cameras() -> Cameras:
    with open("cached_data/cameras_train.pkl", "rb") as f:
        cameras = pickle.load(f)
    return cameras

"""
dataset, cameras = load_test_data(), load_cameras()
#cameras = load_cameras()
s = SimpleViewSelection(dataset=dataset, cameras=cameras, Rmin=5.0, Tmin=1.0)
s.select_views()
print("Hallo")
"""
