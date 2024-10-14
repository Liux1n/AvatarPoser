'''
# --------------------------------------------
# data preprocessing for AMASS dataset
# --------------------------------------------
# AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing (ECCV 2022)
# https://github.com/eth-siplab/AvatarPoser
# Jiaxi Jiang (jiaxi.jiang@inf.ethz.ch)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
'''
import torch
import numpy as np
import os
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.rotation_tools import aa2matrot,matrot2aa,local2global_pose
from utils import utils_transform
import time
import pickle
import json
import random
random.seed(0)

dataset_hps ="/local/home/liuqing/SP/dataset_hps" # root of amass dataset

path_head_camera_localizations = dataset_hps + "/head_camera_localizations"

print(path_head_camera_localizations)

frame_count_dict  = {}
# Iterate over each file in the directory
for file_name in os.listdir(path_head_camera_localizations):
    if file_name.endswith('.json'):
        file_path = os.path.join(path_head_camera_localizations, file_name)
        
        # Load each JSON file and count the frames
        with open(file_path, 'r') as file:
            data = json.load(file)
            frame_count = len(data)
        
        # Store the frame count in the dictionary
        frame_count_dict[file_name] = frame_count


total_frames = sum(frame_count_dict.values())
num_training_set = total_frames * 0.9

sorted_files = sorted(frame_count_dict.items(), key=lambda x: x[1], reverse=True)


cumulative_sum = 0
train_files = []
test_files = []

for file_name, frame_count in sorted_files:
    if cumulative_sum + frame_count <= num_training_set:
        cumulative_sum += frame_count
        train_files.append(file_name)
    else:
        test_files.append(file_name)


# Create dictionaries for the split
split_dict = {
    "train": train_files,
    "test": test_files
}

if not os.path.exists('./data_split_hps'):
    os.makedirs('./data_split_hps')

# read the camera file.
camera_path = dataset_hps + '/head_camera_videos/cameras.json'
print(camera_path)
try:
    with open(camera_path, 'r') as file:
        camera_data = json.load(file)
except:
    print("Camera file not found")

# Load JSON data into a dictionary
camera_ids = ['029756', '029757']
for camera_id in camera_ids:
    print(camera_id)


# Save the split to a file
for phase in ["train","test"]:
    split_file = os.path.join("./data_split_hps", phase + "_split.txt")
    with open(split_file, 'w') as file:
        for file_name in split_dict[phase]:
            # remove the .json extension
            file_name = file_name[:-5]
            file.write(file_name + '\n')
