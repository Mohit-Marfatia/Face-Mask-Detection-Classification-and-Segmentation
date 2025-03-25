import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("data/masked_face_segmentation/1/face_crop/000002_1.jpg")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

original = image_rgb.copy()
blurred = cv2.GaussianBlur(original, (15, 15), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)

h, s, v = cv2.split(hsv)

# For black regions of the mask (low value)
mask_dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))

# For yellow regions (Batman logo)
mask_yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))

# Combine the masks
combined_mask = cv2.bitwise_or(mask_dark, mask_yellow)
# Threshold the L channel to detect the dark mask against skin/background

lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

l, a, b = cv2.split(lab)
_, mask_l = cv2.threshold(l, 100, 255, cv2.THRESH_BINARY_INV)

# Threshold the b channel to detect yellow elements
_, mask_b = cv2.threshold(b, 140, 255, cv2.THRESH_BINARY)

# Combine HSV and LAB masks for a more robust result
final_mask = cv2.bitwise_or(combined_mask, cv2.bitwise_or(mask_l, mask_b))
plt.subplot(144)
plt.imshow(final_mask, cmap='gray')
plt.title('Combined HSV and LAB Mask')
plt.axis('off')
plt.show()

kernel = np.ones((5, 5), np.uint8)
mask_closed = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

mask_dilated = cv2.dilate(mask_opened, kernel, iterations=2)

plt.figure(figsize=(15, 10))
plt.subplot(221)
plt.imshow(final_mask, cmap='gray')
plt.title('Initial Combined Mask')
plt.axis('off')

plt.subplot(222)
plt.imshow(mask_closed, cmap='gray')
plt.title('After Closing')
plt.axis('off')

plt.subplot(223)
plt.imshow(mask_opened, cmap='gray')
plt.title('After Opening')
plt.axis('off')

plt.subplot(224)
plt.imshow(mask_dilated, cmap='gray')
plt.title('Final Cleaned Mask')
plt.axis('off')
plt.show()
