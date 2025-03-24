import cv2
import numpy as np
import os

# Set paths
INPUT_FOLDER = "data/masked_face_segmentation/1/face_crop/"
OUTPUT_MASK_FOLDER = "results/masks/"
OUTPUT_SEGMENTED_FOLDER = "results/segmented/"

# Create output directories
os.makedirs(OUTPUT_MASK_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_SEGMENTED_FOLDER, exist_ok=True)

# Define HSV range for mask segmentation
LOWER_BOUND = np.array([85, 20, 30])
UPPER_BOUND = np.array([160, 255, 255])


def load_images(folder):
    """Returns a list of image file paths from a folder."""
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))]


def preprocess_image(image_path, size=(600, 600)):
    """Loads and resizes an image."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    return cv2.resize(image, size)


def segment_mask(image, lower_bound, upper_bound, kernel_size=(5, 5)):
    """Segments the mask from the image using color thresholding and morphology operations."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    kernel = np.ones(kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_mask, [max_contour], -1, 255, thickness=cv2.FILLED)

    return cv2.GaussianBlur(final_mask, (7, 7), 0)


def save_results(image_name, final_mask, image):
    """Saves the mask and segmented image if it meets the white pixel percentage threshold."""
    white_ratio = np.sum(final_mask == 255) / final_mask.size

    if 0.1 < white_ratio < 0.75:
        segmented = cv2.bitwise_and(image, image, mask=final_mask)
        cv2.imwrite(os.path.join(OUTPUT_MASK_FOLDER, image_name), final_mask)
        cv2.imwrite(os.path.join(OUTPUT_SEGMENTED_FOLDER, image_name), segmented)
        print(f"Saved {image_name}, mask coverage: {white_ratio:.2%}")


def main():
    """Processes all images in the input folder."""
    image_files = load_images(INPUT_FOLDER)

    for img_path in image_files:
        image_name = os.path.basename(img_path)
        image = preprocess_image(img_path)

        if image is None:
            print(f"Skipping {image_name}, unable to read image.")
            continue

        final_mask = segment_mask(image, LOWER_BOUND, UPPER_BOUND)
        save_results(image_name, final_mask, image)

    print("Processing complete.")


if __name__ == "__main__":
    main()
