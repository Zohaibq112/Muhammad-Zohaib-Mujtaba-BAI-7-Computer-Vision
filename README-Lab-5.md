# LAB 05: HOG-Based Industrial Defect Detection and Classification

An automated visual inspection prototype for steel surfaces. It uses HOG (Histogram of Oriented Gradients) features with machine-learning classifiers to decide whether a product should be accepted or rejected.

## Pipeline

```
Image -> Preprocessing -> HOG Feature Extraction -> Classifier (SVM / Random Forest) -> ACCEPT / REJECT
```

## Dataset

- **Name:** NEU Steel Surface Defect (train/valid split) on Kaggle
- **Kaggle slug:** `sovitrath/neu-steel-surface-defect-detect-trainvalid-split`
- **Content:** 1,800 grayscale images (200x200) with XML bounding boxes
- **Classes:** crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches

**Important:** the dataset has no "normal" images. For binary classification, the code cuts 100x100 patches from each image using the XML boxes:

| Patch type | Label |
|---|---|
| No overlap with any defect box | 0 (Non-Defective) |
| At least 40% overlap with a defect box | 1 (Defective) |

Patches are resized to 64x64 and the classes are balanced. Train, validation and test splits are made by source image, so patches from one image never appear in two splits.

For 6-class classification, set `MODE = "multiclass"` in the dataset cell. Full images are resized to 128x128.

## How to run (Google Colab)

1. Open `LAB05_HOG_Defect_Detection.ipynb` in Google Colab.
2. Run the cells from top to bottom (Runtime -> Run all).
3. If the Kaggle download asks for login, add a Kaggle API token as a Colab secret, or download the zip manually, unzip it and set `path` to that folder.

## What each cell does

| Cell | Purpose |
|---|---|
| 1 | Install `kagglehub` and download the dataset |
| 2 | Load and inspect the data, show one sample per class |
| 3 | Preprocess (grayscale, resize, histogram equalisation) and split the data |
| 4 | Extract HOG features and visualise them |
| 5 | Train HOG + SVM and HOG + Random Forest, print the confusion matrix, classification report, accuracy, precision, recall and F1 |
| 6 | HOG parameter study: cell sizes 4, 8, 16 and orientations 6, 9, 12. Best setting chosen on validation F1 |
| 7 | Robustness test: brightness, Gaussian noise, rotation, blur. Reports the change in accuracy and F1 |
| 8 | Final decision module `inspect_product()` printing the inspection result |
| 9 | (Optional) Upload and inspect your own image |

## Example output

```
PRODUCT INSPECTION RESULT
Prediction: DEFECTIVE
Confidence: 93.0%
Action: REJECT PRODUCT
```

or

```
PRODUCT INSPECTION RESULT
Prediction: NON-DEFECTIVE
Confidence: 88.0%
Action: ACCEPT PRODUCT
```

In binary mode the module scans the surface with 100x100 windows (stride 50). The product is rejected if any window is classified as defective.

## Settings you can change

| Setting | Where | Meaning |
|---|---|---|
| `MODE` | Cell 3 | `"binary"` or `"multiclass"` |
| `IMG` | Cell 3 | Common image resolution |
| `P` | Cell 3 | Patch size for binary mode |
| `reject_threshold` | Cell 8 | Defect probability above which a product is rejected |

## Report mapping

- **Results:** tables and plots from Cells 5 and 6
- **Robustness Analysis:** table and bar chart from Cell 7
- **Industrial Deployment:** the decision output from Cell 8
- **Limitations:** the normal class is built from patches, the dataset has a single steel type, and images are captured under controlled conditions

## Bonus (real-time)

Colab cannot access a local webcam. To build the real-time prototype, run the `inspect_product` logic inside a `cv2.VideoCapture(0)` loop on a local machine, or process the frames of an uploaded video, and draw PASS or DEFECTIVE on each frame with `cv2.putText`.

## Requirements

Python 3, `kagglehub`, `numpy`, `opencv-python`, `scikit-image`, `scikit-learn`, `pandas`, `matplotlib` (all except `kagglehub` are preinstalled in Colab).
