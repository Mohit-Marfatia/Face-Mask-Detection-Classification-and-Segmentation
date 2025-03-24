# Face Mask Detection and Segmentation Project

## Introduction
This project aims to develop a computer vision solution for classifying and segmenting face masks in images. The objective is to implement both traditional and deep learning techniques for mask detection and segmentation.

## Dataset
- **Source**: 
  - Face Mask Detection Dataset: [GitHub Repository](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset)
  - Masked Face Segmentation Dataset: [GitHub Repository](https://github.com/sadjadrz/MFSD)
- **Structure**: Contains images of people with and without face masks, along with ground truth segmentation masks

## Methodology for Traditional Segmentation (Part C)
## Experimental Approaches

### HSV and LAB Color Space Combination
**File**: `combine_hsv_lab.py`

We explored a more complex color-based segmentation approach that combined HSV and LAB color spaces:

#### Methodology:
- Convert image to HSV and LAB color spaces
- Create masks using:
  - Dark regions in HSV (for black mask areas)
  - Yellow regions in HSV (for logos)
  - L channel thresholding
  - B channel thresholding
- Combine multiple color space masks
- Apply morphological operations (closing, opening, dilation)

#### Challenges:
- Multiple color space masks led to complex segmentation
- Increased computational complexity
- Less consistent results compared to single HSV approach

### Haar Cascade and K-Means Clustering
**File**: `haar_cascade_and_kmeans.py`

An alternative approach combining face detection and image segmentation:

#### Technique:
- Use Haar cascade classifier for face detection
- Apply K-Means clustering for image segmentation
- Extract lower face regions

#### Limitations:
- Inconsistent face detection
- K-Means clustering not specifically tailored to mask segmentation
- High computational overhead

### Image Preprocessing
**File**: `resize_and_contrast_equalze.py`

Explored image preprocessing techniques to improve segmentation:

#### Preprocessing Steps:
- Resize images with padding
- Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Maintain aspect ratio during resizing

#### Benefits:
- Standardized image size
- Enhanced local image contrast
- Improved feature visibility

## Image Segmentation Approach
We developed a mask segmentation method using color-based thresholding and morphological operations:

1. **Color Space Conversion**: 
   - Converted images to HSV color space for better color-based segmentation
   - Used interactive HSV tuner to find optimal color range for mask detection

2. **Segmentation Techniques**:
   - Applied color thresholding using custom HSV range
   - Used morphological operations (closing and opening) to refine the mask
   - Selected the largest contour to create the final mask

### HSV Color Range Tuning
- Developed an interactive HSV tuner (`hsv_tuner.py`) to manually adjust color thresholds
- Final HSV range used: 
  - Lower Bound: `[85, 20, 30]`
  - Upper Bound: `[160, 255, 255]`

  ![HSV Tuner 1](screenshots/hsv_tuner_1.png) 
  ![HSV Tuner 2](screenshots/hsv_tuner_2.png) 

### Segmentation Process
1. Preprocess images by resizing
2. Convert to HSV color space
3. Create binary mask using color thresholding
4. Apply morphological operations
5. Extract largest contour as final mask
6. Only save those images whose white pixel percentage threshold of being in between 10% and 75%.

## Performance Evaluation

### Intersection over Union (IoU) Analysis
- Computed IoU between predicted and ground truth masks
- **Average IoU**: 0.5719 (57.19%)

  ![Average IoU](screenshots/average_iou.png) 

### Visualization
Included sample images showing:
- Original images
- Extracted masks
- Segmented regions

## Key Challenges and Solutions
- Varying mask colors and materials
- Inconsistent lighting conditions
- Handling different mask styles

## How to Run the Code

### Prerequisites
- Python 3.7+

### Installation
1. Clone the repository
```bash
git clone https://github.com/yourusername/VR_Project1_YourName_YourRollNo.git
cd VR_Project1_YourName_YourRollNo
 ```

2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
3. Install dependencies

```bash
pip install -r requirements.txt
```
### Execution
Run the segmentation script:

```bash
python src/segmentation.py
```