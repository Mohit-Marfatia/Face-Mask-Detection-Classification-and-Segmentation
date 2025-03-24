import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def detect_lower_face_and_segment(folder_path, output_folder, max_images=500, k_clusters=2):
    # Load Haar cascade for lower face detection
    lower_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if lower_face_cascade.empty():
        print("Error loading Haar cascade.")
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get list of image files and limit to max_images
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))][:max_images]
    
    for filename in image_files:
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            continue

        lower_face_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Reshape image for clustering
        pixel_values = lower_face_rgb.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        
        # Apply K-Means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, k_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert centers to integer values
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(lower_face_rgb.shape)
            
        gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
        lower_faces = lower_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in lower_faces:
            lower_face = segmented_image[y:y+h, x:x+w]
            
           
            # Convert segmented image back to BGR and save
            output_path = os.path.join(output_folder, f"segmented_{filename}")
            cv2.imwrite(output_path, cv2.cvtColor(lower_face, cv2.COLOR_RGB2BGR))
            

# Example usage
folder_path = "data/masked_face_segmentation/1/face_crop"
output_folder = "results/masks"
detect_lower_face_and_segment(folder_path, output_folder, max_images=100)
