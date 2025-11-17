import os
import cv2
import numpy as np
from detection import AccidentDetectionModel
from pathlib import Path
import csv

# Paths
accident_dir = Path("data/test/Accident")
nonaccident_dir = Path("data/test/Non Accident")

# Model
model = AccidentDetectionModel("model.json", "model_weights.h5")

results = []

def test_folder(folder, label):
    correct = 0
    total = 0
    for imgfile in folder.glob("*.jpg"):
        img = cv2.imread(str(imgfile))
        if img is None:
            print(f"Could not read {imgfile}")
            continue
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(rgb_img, (250, 250))
        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        pred_label = 1 if pred == "Accident" else 0
        gt_label = 1 if label == "Accident" else 0
        prob_val = float(prob[0][0])
        results.append({
            "file": str(imgfile),
            "ground_truth": label,
            "predicted": pred,
            "probability": prob_val
        })
        if pred_label == gt_label:
            correct += 1
        total += 1
    return correct, total

acc_correct, acc_total = test_folder(accident_dir, "Accident")
nonacc_correct, nonacc_total = test_folder(nonaccident_dir, "No Accident")

total_correct = acc_correct + nonacc_correct
total = acc_total + nonacc_total

print(f"Accident: {acc_correct}/{acc_total} correct")
print(f"Non Accident: {nonacc_correct}/{nonacc_total} correct")
print(f"Total: {total_correct}/{total} correct")

# Optionally write results to CSV
with open("test_image_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "ground_truth", "predicted", "probability"])
    writer.writeheader()
    writer.writerows(results)
print("Results saved to test_image_results.csv")
