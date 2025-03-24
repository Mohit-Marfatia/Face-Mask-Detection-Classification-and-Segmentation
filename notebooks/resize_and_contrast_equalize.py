import cv2
import numpy as np
import os

def resize_with_padding(img, target_size=(256, 256), pad_color=(0, 0, 0)):
    h, w = img.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_img = cv2.resize(img, (new_w, new_h))

    padded_img = np.full((target_size[1], target_size[0], 3), pad_color, dtype=np.uint8)

    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img

    return padded_img

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # Convert to LAB color space
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

def process_images(input_folder, output_folder, target_size=(256, 256)):
    os.makedirs(output_folder, exist_ok=True)

    resize_folder=os.path.join(output_folder, "resize")
    contrast_equalize_folder=os.path.join(output_folder, "contrast_equalize")

    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is not None:
            processed_img = resize_with_padding(img, target_size)
            cv2.imwrite(os.path.join(resize_folder, filename), processed_img)
            clahe_img = apply_clahe(processed_img)
            cv2.imwrite(os.path.join(contrast_equalize_folder, filename), clahe_img)

process_images("data/masked_face_segmentation/1/face_crop", "processed_data")


