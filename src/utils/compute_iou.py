import cv2
import numpy as np
import os

def calculate_iou(predicted_mask, ground_truth_mask):
    """
    Compute the Intersection over Union (IoU) between predicted and ground truth masks.
    """
    predicted_mask = predicted_mask > 0  # Convert to binary
    ground_truth_mask = ground_truth_mask > 0

    intersection = np.logical_and(predicted_mask, ground_truth_mask)
    union = np.logical_or(predicted_mask, ground_truth_mask)

    return np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0

def compute_iou_for_folder(predicted_mask_folder, ground_truth_folder):
    """
    Compute IoU for all masks in the given folders.
    """
    mask_files = [f for f in os.listdir(predicted_mask_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    ious = []

    for mask_name in mask_files:
        pred_path = os.path.join(predicted_mask_folder, mask_name)
        gt_path = os.path.join(ground_truth_folder, mask_name)

        if not os.path.exists(gt_path):
            print(f"Skipping {mask_name}, ground truth not found.")
            continue

        predicted_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        ground_truth_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if predicted_mask is None or ground_truth_mask is None:
            print(f"Skipping {mask_name}, unable to read mask images.")
            continue

        ground_truth_mask = cv2.resize(ground_truth_mask, (predicted_mask.shape[1], predicted_mask.shape[0]))

        iou = calculate_iou(predicted_mask, ground_truth_mask)
        ious.append(iou)

    avg_iou = sum(ious) / len(ious) if ious else 0
    return avg_iou

if __name__ == "__main__":
    predicted_mask_folder = "results/masks"
    ground_truth_folder = "data/masked_face_segmentation/1/face_crop_segmentation"

    avg_iou = compute_iou_for_folder(predicted_mask_folder, ground_truth_folder)
    print(f"\nAverage IoU: {avg_iou:.4f}")
