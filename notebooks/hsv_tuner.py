import cv2
import numpy as np

# Load image
image = cv2.imread("data/masked_face_segmentation/1/face_crop/000000_1.jpg") 
image = cv2.resize(image, (600, 600))  # Resize for better visualization
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Callback function for trackbars
def nothing(x):
    pass

# Create a window
cv2.namedWindow("Trackbars")

# Create trackbars for lower and upper HSV values
cv2.createTrackbar("Lower-H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("Lower-S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Lower-V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Upper-H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("Upper-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Upper-V", "Trackbars", 255, 255, nothing)

while True:
    # Get trackbar values
    l_h = cv2.getTrackbarPos("Lower-H", "Trackbars")
    l_s = cv2.getTrackbarPos("Lower-S", "Trackbars")
    l_v = cv2.getTrackbarPos("Lower-V", "Trackbars")
    u_h = cv2.getTrackbarPos("Upper-H", "Trackbars")
    u_s = cv2.getTrackbarPos("Upper-S", "Trackbars")
    u_v = cv2.getTrackbarPos("Upper-V", "Trackbars")

    # Define lower and upper HSV range
    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    # Apply threshold
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Show results
    cv2.imshow("Original", image)
    cv2.imshow("Mask", mask)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
print(f"Lower HSV: {lower_bound}, Upper HSV: {upper_bound}")
