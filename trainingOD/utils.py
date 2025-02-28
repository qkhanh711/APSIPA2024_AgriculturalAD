import cv2 as cv
import os
import matplotlib.pyplot as plt
import json
import numpy as np
from tqdm import tqdm

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area

def draw_boxes_and_save_to_json(folder, img_path, txt_paths, json_path, index, cls, show = False, save = True):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    height, width, _ = img.shape

    colors = [(0, 255, 0), (255, 0, 0)]
    bounding_boxes = {}

    box_data = {'labels': [], 'predict': []}

    for txt_path, color in zip(txt_paths, colors):
        label_type = "labels" if "train/labels" in txt_path or "test/labels" in txt_path else "predict"
        with open(txt_path, "r") as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                _, x_center, y_center, w, h = map(float, parts)
                x_center *= width
                y_center *= height
                w *= width
                h *= height

                x1 = int((x_center * 2 - w) / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)

                box_data[label_type].append((i, [x1, y1, x2, y2]))

                if show:
                    cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

    for train_idx, train_box in box_data['labels']:
        best_iou = 0
        best_predict_idx = -1
        for predict_idx, predict_box in box_data['predict']:
            current_iou = iou(train_box, predict_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_predict_idx = predict_idx
        
        
        if best_predict_idx != -1:
            x1, y1, x2, y2 = train_box
            x1_p, y1_p, x2_p, y2_p = box_data['predict'][best_predict_idx][1]
            saved_path = f"{folder}/annotations/{cls}/img/{len(os.listdir(f'{folder}/annotations/{cls}/img')):04d}.png"
            bounding_boxes[saved_path] = {
                'labels': [x1, y1, x2 - x1, y2 - y1],
                'predict': [x1_p, y1_p, x2_p - x1_p, y2_p - y1_p],
                'IoU' : best_iou
            }
            
            obj_predict_img = img[y1_p:y2_p, x1_p:x2_p]
            if save:
                cv.imwrite(saved_path, cv.cvtColor(obj_predict_img, cv.COLOR_RGB2BGR))

    if show:
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    if not show:
        with open(json_path, 'w') as json_file:
            json.dump(bounding_boxes, json_file, indent=4)


def list_images(folder, cls, f):
    count = 0
    json_files = []
    import re
    index = [int(re.findall(r'\d+', i)[0]) for i in os.listdir(f"{folder}/annotations/{cls}/anot")]
    for i in index:
        with open(f"{folder}/annotations/{cls}/anot/{i}.json") as json_file:
            data = json.load(json_file)
            for key in data:
                if data[key]['IoU'] > f:
                    count += 1
                    # print(key)
                    json_files.append(key)
    print(count)
    return json_files

def Anotate(folder, indexing, cls, show = False):
    for index in tqdm(indexing):
        img_path = f"{folder}/images/{cls}_{index}.png"
        txt_path1 = f"{folder}/labels/{cls}_{index}.txt"
        txt_path2 = f"{folder}/predict/{cls}_{index}.txt"
        json_path = f"{folder}/annotations/{cls}/anot/{index}.json"

        txt_paths = [txt_path1, txt_path2]
        os.makedirs(f"{folder}/annotations/{cls}/img", exist_ok=True)
        os.makedirs(f"{folder}/annotations/{cls}/anot", exist_ok=True)
        if txt_paths:
            try:
                draw_boxes_and_save_to_json(folder, img_path, txt_paths, json_path, index, cls, show)
            except Exception as e:
                print(f"Failed to process {img_path}")

def load_annotations(folder_path):
    annotations = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    annotations[file_path] = data
    return annotations

def calculate_iou(pred_box, gt_box):
    # Determine the coordinates of the intersection rectangle
    x_left = max(pred_box[0], gt_box[0])
    y_top = max(pred_box[1], gt_box[1])
    x_right = min(pred_box[2], gt_box[2])
    y_bottom = min(pred_box[3], gt_box[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both the prediction and ground truth rectangles
    pred_box_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    # Calculate the intersection over union
    iou = intersection_area / float(pred_box_area + gt_box_area - intersection_area)
    return iou

def calculate_ap(recalls, precisions):
    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls = np.array(recalls)[sorted_indices]
    precisions = np.array(precisions)[sorted_indices]
    
    # Add (0,1) point for precision-recall curve
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))
    
    # Ensure precision is monotonically decreasing
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    
    # Compute the AP
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap

def calculate_map(iou_threshold, annotations):
    aps = []
    for key, value in annotations.items():
        tp = []
        fp = []
        matched_gt_boxes = 0
        for k, v in value.items():
            iou = v["IoU"]
            if iou >= iou_threshold:
                tp.append(1)
                fp.append(0)
                matched_gt_boxes += 1
            else:
                tp.append(0)
                fp.append(1)

        if matched_gt_boxes == 0:
            aps.append(0.0)
            continue

        tp = np.array(tp)
        fp = np.array(fp)

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        recalls = tp_cumsum / matched_gt_boxes

        ap = calculate_ap(recalls, precisions)
        aps.append(ap)
    return np.mean(aps)