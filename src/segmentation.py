import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from notebooks import segment_mask
from src.utils.compute_iou import compute_iou_for_folder

segment_mask.main()

predicted_mask_folder = "results/masks"
ground_truth_folder = "data/masked_face_segmentation/1/face_crop_segmentation"

avg_iou = compute_iou_for_folder(predicted_mask_folder, ground_truth_folder)
print(f"\nAverage IoU from segmentation.py: {avg_iou:.4f}")