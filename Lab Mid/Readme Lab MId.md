HOG vs SIFT on Blood-Cell Images
Computer Vision Lab – Mid Exam

Name: M. Zohaib
Roll No: 040
Course: Computer Vision Lab

1. Project Overview

This project compares two traditional computer vision feature extraction techniques, Histogram of Oriented Gradients (HOG) and Scale-Invariant Feature Transform (SIFT), for blood-cell image classification.

A Linear Support Vector Machine (SVM) is used with both feature extraction methods to classify four types of blood cells:

Eosinophil
Monocyte
Neutrophil
Platelet

The project also evaluates model robustness under image rotation and Gaussian noise.

2. Dataset

The project uses the BloodMNIST dataset from MedMNIST.

Parameter	Value
Image size	128 × 128 pixels
Image format	Grayscale
Training images	400
Validation images	120
Testing images	120
Number of classes	4
Training images per class	100
Validation images per class	30
Testing images per class	30
3. Personal Configuration
Parameter	Value
Registration number	040
Random seed	40
HOG cell size	8 × 8
SIFT grid size	4 × 4
Number of SIFT keypoints	16
HOG feature length	8100
SIFT feature length	2048
4. Technologies Used
Python
Google Colab
OpenCV
NumPy
Pandas
Matplotlib
Scikit-learn
Scikit-image
MedMNIST
5. Methodology
5.1 Data Preparation

BloodMNIST images are loaded and divided into training, validation and testing sets. Images are converted to grayscale and resized to 128 × 128 pixels.

5.2 HOG Feature Extraction

HOG extracts gradient orientation information from images.

Configuration:

Orientations: 9
Pixels per cell: 8 × 8
Cells per block: 2 × 2
Block normalization: L2-Hys
Feature length: 8100

Feature length calculation:

Cells per side = 128 / 8 = 16
Blocks = 15 × 15 = 225
Values per block = 2 × 2 × 9 = 36
Total features = 225 × 36 = 8100
5.3 SIFT Feature Extraction

SIFT extracts local image descriptors. A fixed 4 × 4 grid is used to ensure that every image has the same feature-vector length.

Configuration:

Grid size: 4 × 4
Number of keypoints: 16
Descriptor length: 128
Total feature length: 16 × 128 = 2048
5.4 SVM Classification

A Linear SVM is trained using both HOG and SIFT features.

The following C values are tested:

0.1
1
10

The best C value is selected using validation accuracy.

5.5 Robustness Testing

Both models are evaluated under three conditions:

Clean images.
Rotated images (20 degrees).
Noisy images (Gaussian noise, sigma = 0.05).
6. Experimental Results

The following values are illustrative predictions, not actual measured results. They must be replaced with the outputs from Google Colab.

Model	Clean Accuracy	Rotated Accuracy	Noisy Accuracy
HOG + SVM	77%	62%	70%
SIFT + SVM	67%	65%	55%
Feature Comparison
Feature	HOG	SIFT
Feature length	8100	2048
Feature type	Gradient histograms	Local descriptors
Grid	8 × 8 cells	4 × 4 keypoints
Rotation robustness	Expected to be lower	Some rotation robustness
Classifier	Linear SVM	Linear SVM
7. Observations
HOG captures image shape and gradient information.
SIFT extracts local descriptors from selected keypoints.
HOG produces a longer feature vector than fixed-grid SIFT.
Fixed-grid SIFT ensures a consistent feature length for every image.
Rotation and noise can affect the classification performance of both methods.
Grayscale conversion removes stain-colour information, which can make some blood-cell classes harder to distinguish.
8. Conclusion

This project compares HOG and SIFT feature extraction techniques for blood-cell image classification using Linear SVM.

HOG represents images through gradient orientation histograms, while SIFT represents local image patterns using keypoint descriptors.

Both methods provide different approaches to extracting visual information. The robustness experiment helps examine how rotation and noise influence classification performance.

The final comparison should be based on the actual experimental results obtained from Google Colab.

9. Project Structure
text
Blood-Cell-HOG-SIFT/
│
├── RollNo_Name_LabMid.ipynb
├── README.md
├── 040_M_Zohaib_LabMid.md
│
└── images/
    ├── q1_grid.png
    ├── q2_hog.png
    ├── q2_cm.png
    ├── q3_kp.png
    └── q3_cm.png

10. How to Run
Open the notebook in Google Colab.
Install the required package:
python
!pip install -q -U medmnist

Run all notebook cells in order.
Allow the BloodMNIST dataset to download.
Check the printed feature lengths, validation scores and confusion matrices.
Run the robustness evaluation.
Save the notebook with all outputs visible.
11. Submission Details

Student Name: M. Zohaib
Roll Number: 040
Course: Computer Vision Lab
Project Title: HOG vs SIFT on Blood-Cell Images

Submitted as part of the Computer Vision Lab Mid Exam.
