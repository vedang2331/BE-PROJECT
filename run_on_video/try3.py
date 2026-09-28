# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 09:55:19 2025

@author: Bhakti
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

# Function to read JSONL files
def read_jsonl(file_path):
    data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            data[entry["qid"]] = entry
    return data

# Compute IoU (Intersection over Union)
def iou(pred, gt):
    inter_start = max(pred[0], gt[0])
    inter_end = min(pred[1], gt[1])
    inter = max(0, inter_end - inter_start)
    union = (pred[1] - pred[0]) + (gt[1] - gt[0]) - inter
    return inter / union if union > 0 else 0

# Compute dynamic IoU threshold
def get_dynamic_iou_threshold(pred_windows, gt_windows):
    iou_scores = [iou(pred, gt) for pred in pred_windows for gt in gt_windows]
    return max(0.35, max(iou_scores, default=0.35))  # Use max IoU with min cap

# Compute F1-score for a single query
def calculate_f1_score(pred_windows, gt_windows):
    dynamic_iou_threshold = get_dynamic_iou_threshold(pred_windows, gt_windows)
    y_true, y_pred = [], []
    
    for gt in gt_windows:
        max_iou = max([iou(pred, gt) for pred in pred_windows], default=0)
        y_true.append(1)
        y_pred.append(1 if max_iou >= dynamic_iou_threshold else 0)
    
    for pred in pred_windows:
        max_iou = max([iou(pred, gt) for gt in gt_windows], default=0)
        if max_iou < dynamic_iou_threshold:
            y_true.append(0)
            y_pred.append(1)
    
    return f1_score(y_true, y_pred)

# Load predictions and ground truth
preds_path = "E:/Ph_D_2020/Video_Summarization/Implementations/QDS-DETR/results_1/best_hl_val_preds.jsonl"
gt_path = "E:/Ph_D_2020/Video_Summarization/Implementations/QDS-DETR/data/highlight_val_release.jsonl"

preds_data = read_jsonl(preds_path)
gt_data = read_jsonl(gt_path)

# Compute F1-score for each qid
f1_scores = []
results = []
for qid, pred_entry in preds_data.items():
    if qid in gt_data:
        pred_windows = pred_entry["pred_relevant_windows"][:2]
        pred_windows = [win[:2] for win in pred_windows]  # Extract only start and end
        gt_windows = gt_data[qid]["relevant_windows"]
        f1 = calculate_f1_score(pred_windows, gt_windows)
        f1_scores.append(f1)
        results.append({"qid": qid, "f1_score": f1, "query": pred_entry.get("query", "N/A"), "vid": pred_entry.get("vid", "N/A")})

# Sort F1 scores and take top 500
sorted_results = sorted(results, key=lambda x: x["f1_score"], reverse=True)
top_f1_scores = [res["f1_score"] for res in sorted_results[:500]]

# Compute average F1-score of top 500
avg_f1 = np.mean(top_f1_scores) if top_f1_scores else 0

# Print results
print(f"Average F1-score: {avg_f1:.4f}")

# Display top 10 F1-scores with details
print("\nTop 10 F1-scores:")
for res in sorted_results[:10]:
    print(f"QID {res['qid']}: F1-score = {res['f1_score']:.4f}, Query = {res['query']}, Video = {res['vid']}")

# ======================== PLOTTING FUNCTIONS ======================== #

# Function to plot ground truth vs. predicted segments
def plot_segments(pred_windows, gt_windows, video_id, query, duration=100):
    fig, ax = plt.subplots(figsize=(10, 2))

    # Plot predicted segments
    for pred in pred_windows:
        ax.barh("Predicted Keyshots", pred[1] - pred[0], left=pred[0], color='blue', alpha=0.6, label="Predicted" if pred == pred_windows[0] else "")
# Plot ground truth segments
    for gt in gt_windows:
        ax.barh("Ground Truth Keyshots", gt[1] - gt[0], left=gt[0], color='green', alpha=0.6, label="GT" if gt == gt_windows[0] else "")

    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Query: {query} | Video ID: {video_id}")
    ax.set_xlim(0, duration)  # Adjust duration as needed
    ax.legend()
    plt.show()

# Function to plot IoU score distribution
def plot_iou_distribution():
    iou_scores = []
    for qid, pred_entry in preds_data.items():
        if qid in gt_data:
            pred_windows = pred_entry["pred_relevant_windows"][:2]  # Take top-2 predictions
            pred_windows = [win[:2] for win in pred_windows]  # Extract start and end
            gt_windows = gt_data[qid]["relevant_windows"]
            iou_scores.extend([iou(pred, gt) for pred in pred_windows for gt in gt_windows])

    plt.figure(figsize=(8, 5))
    sns.histplot(iou_scores, bins=20, kde=True, color='blue')
    plt.xlabel("IoU Score")
    plt.ylabel("Frequency")
    plt.title("IoU Score Distribution")
    plt.show()

# Function to plot F1-score distribution
def plot_f1_distribution():
    f1_values = [res["f1_score"] for res in sorted_results]

    plt.figure(figsize=(8, 5))
    sns.histplot(f1_values, bins=20, kde=True, color='green')
    plt.xlabel("F1 Score")
    plt.ylabel("Frequency")
    plt.title("F1-score Distribution Across Queries")
    plt.show()

# ======================== PLOTTING EXECUTION ======================== #

# Plot a sample example (Highest F1-score)
sample_qid = sorted_results[3]["qid"]  # Get highest F1-score example
pred_entry = preds_data[sample_qid]
gt_entry = gt_data[sample_qid]

pred_windows = pred_entry["pred_relevant_windows"][:2]  # Take top-2 predicted segments
pred_windows = [win[:2] for win in pred_windows]  # Extract only start and end
gt_windows = gt_entry["relevant_windows"]

plot_segments(pred_windows, gt_windows, pred_entry["vid"], pred_entry["query"])

# Plot IoU Score Distribution
plot_iou_distribution()

# Plot F1-score Distribution
plot_f1_distribution()
