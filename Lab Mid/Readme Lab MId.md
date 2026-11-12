HOG vs SIFT on Blood-Cell Images
Computer Vision Lab – Mid Exam
Student Name: M. Zohaib
Roll Number: 040
Course: Computer Vision Lab

Project Overview
This project compares two traditional computer vision feature extraction techniques, Histogram of Oriented Gradients (HOG) and Scale-Invariant Feature Transform (SIFT), for blood-cell image classification.

A Linear Support Vector Machine (SVM) is used with both feature extraction methods to classify four types of blood cells:

Eosinophil
Monocyte
Neutrophil
Platelet
The project also evaluates model robustness under image rotation and Gaussian noise.

Dataset
The project uses the BloodMNIST dataset from MedMNIST.

Image size: 128 × 128 pixels
Image format: Grayscale
Training images: 400
Validation images: 120
Testing images: 120
Classes: 4
Images per training class: 100
Images per validation/test class: 30
Personal Configuration

Parameter	Value
Registration number	040
Random seed	40
HOG cell size	8 × 8
SIFT grid size	4 × 4
Number of SIFT keypoints	16
HOG feature length	8100
SIFT feature length	2048
Technologies Used
Python
Google Colab
OpenCV
NumPy
Pandas
Matplotlib
Scikit-learn
Scikit-image
MedMNIST
Methodology
1. Data Preparation
BloodMNIST images are loaded and divided into training, validation and testing sets. Images are converted to grayscale and resized to 128 × 128 pixels.

2. HOG Feature Extraction
HOG extracts gradient orientation information from images.

Configuration:

Orientations: 9
Pixels per cell: 8 × 8
Cells per block: 2 × 2
Block normalization: L2-Hys
Feature length: 8100
3. SIFT Feature Extraction
SIFT extracts local image descriptors. A fixed 4 × 4 grid is used to ensure that each image has the same feature-vector length.

Configuration:

Grid size: 4 × 4
Keypoints: 16
Descriptor length: 128
Total feature length: 2048
4. SVM Classification
A Linear SVM is trained using both HOG and SIFT features.

The following C values are tested:

0.1
1
10
The best C value is selected using validation accuracy.

5. Robustness Testing
Both models are evaluated under three conditions:

Clean images
Rotated images (20 degrees)
Noisy images (Gaussian noise, sigma = 0.05)
Experimental Results
The following are preliminary illustrative estimates, not verified experimental results. Replace them with the actual Google Colab outputs before submission.


Model	Estimated Clean Accuracy	Estimated Rotated Accuracy	Estimated Noisy Accuracy
HOG + SVM	77%	62%	70%
SIFT + SVM	67%	65%	55%
Feature Comparison

Feature	HOG	SIFT
Feature length	8100	2048
Feature type	Gradient histograms	Local descriptors
Grid	8 × 8 cells	4 × 4 keypoints
Rotation robustness	Lower expected	Some rotation robustness
Classification	Linear SVM	Linear SVM
Observations
