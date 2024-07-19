import os
import shutil
from tqdm import tqdm
import json
from utils import list_images

import argparse


parser = argparse.ArgumentParser(description='Move images to the target directory')
parser.add_argument('--f', type=float, default=0.75, help='IoU score threshold')
args = parser.parse_args()


folders = ["train_test_splited_data/train", "train_test_splited_data/test"]
cls = ['good', 'tear']

# list images have IoU score more than f%

threshold = [0.5, 0.75, 0.9]


for folder in folders:
    print(f"Folder: {folder}")
    for i in threshold:
        print(f"IoU score > {i}")
        print("Good")
        list_images(folder, cls[0], i)
        print("Tear")
        list_images(folder, cls[1], i)

f = args.f
print(f)
good_train_images = list_images(folders[0], cls[0], f)
tear_train_images = list_images(folders[0], cls[1], f)
good_test_images = list_images(folders[1], cls[0], f)
tear_test_images = list_images(folders[1], cls[1], f)

def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def move(folder, cls, images):
    print(folder[-5:], cls)
    # Determine the target directory based on folder and class
    if folder[-5:] == "train" and cls == "good":
        target_dir = "../../../patchcore-inspection/mvtec_anomaly_detection/bean/train/good"
    elif folder[-5:] == "train" and cls == "tear":
        target_dir = "../../../patchcore-inspection/mvtec_anomaly_detection/bean/test/tear"
    elif folder[-4:] == "test" and cls == "good":
        target_dir = "../../../patchcore-inspection/mvtec_anomaly_detection/bean/test/good"
    else:
        target_dir = "../../../patchcore-inspection/mvtec_anomaly_detection/bean/test/tear"
    
    # Create the target directory if it doesn't exist
    create_dir_if_not_exists(target_dir)
    
    # Copy images to the target directory
    for image in tqdm(images):
        shutil.copy(image, target_dir)

# Example usage
move(folders[0], cls[0], good_train_images)
move(folders[0], cls[1], tear_train_images)
move(folders[1], cls[0], good_test_images)
move(folders[1], cls[1], tear_test_images)
