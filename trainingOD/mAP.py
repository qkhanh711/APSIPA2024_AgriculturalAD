# %%
import os
import json
import numpy as np
from utils import calculate_map, load_annotations

# Paths to the annotation directories
test_good_path = 'train_test_splited_data/test/annotations/good/anot'
test_tear_path = 'train_test_splited_data/test/annotations/tear/anot'
train_good_path = 'train_test_splited_data/train/annotations/good/anot'
train_tear_path = 'train_test_splited_data/train/annotations/tear/anot'

# Load annotations
test_good_annotations = load_annotations(test_good_path)
test_tear_annotations = load_annotations(test_tear_path)
train_good_annotations = load_annotations(train_good_path)
train_tear_annotations = load_annotations(train_tear_path)

# Calculate AP for each category
iou_thresholds = [0.5, 0.75, 0.90]

for iou_threshold in iou_thresholds:
    ap_test_good = calculate_map(iou_threshold, test_good_annotations)
    ap_test_tear = calculate_map(iou_threshold, test_tear_annotations)
    ap_train_good = calculate_map(iou_threshold, train_good_annotations)
    ap_train_tear = calculate_map(iou_threshold, train_tear_annotations)
    print(f'AP@{iou_threshold}: {ap_test_good}, {ap_test_tear}, {ap_train_good}, {ap_train_tear}')

    # Calculate MAP
    map_value = np.mean([ap_test_good, ap_test_tear, ap_train_good, ap_train_tear])
    print(f'MAP: {map_value}')



