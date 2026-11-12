# Lab Mid Exam: HOG vs SIFT on Blood-Cell Images

**Name:** ______________  **Roll No:** 040  **Course:** Computer Vision Lab

> How to use this file: run the code in Colab, then paste your real printed output into each **Output** box. Everything marked `___` must be filled from your own results. Do not submit this file with blanks.

---

## Personal Settings

| Setting | Rule | My value |
|---|---|---|
| Random seed | Last two digits of reg. no. (040) | **40** |
| HOG cell size | Last digit 0 is even | **8 x 8** |
| SIFT grid size | Last digit 0 is in 0 to 4 | **4 x 4 (16 keypoints)** |

```python
!pip install -q -U medmnist

REG = "040"
SEED = int(REG[-2:])
d = int(REG[-1])
HOG_CELL = 8 if d % 2 == 0 else 16
GRID = 4 if d <= 4 else 6
print(SEED, HOG_CELL, GRID)
```

**Output:** (expected `40 8 4`)
```
PASTE OUTPUT HERE
```

---

## Q1. Data setup (3 marks)

```python
import numpy as np, cv2, pandas as pd
import matplotlib.pyplot as plt
from medmnist import BloodMNIST

KEEP = {1: 'eosinophil', 5: 'monocyte', 6: 'neutrophil', 7: 'platelet'}
NAMES = list(KEEP.values())
rng = np.random.RandomState(SEED)

def load_split(split, n):
    ds = BloodMNIST(split=split, download=True, size=64)
    X, y = ds.imgs, ds.labels.ravel()
    imgs, labels = [], []
    for new, old in enumerate(KEEP):
        pick = rng.choice(np.where(y == old)[0], n, replace=False)
        for i in pick:
            gray = cv2.cvtColor(X[i], cv2.COLOR_RGB2GRAY)
            imgs.append(cv2.resize(gray, (128, 128)))
            labels.append(new)
    return np.array(imgs), np.array(labels)

Xtr, ytr = load_split('train', 100)
Xva, yva = load_split('val', 30)
Xte, yte = load_split('test', 30)   # not used until Q4

for name, X, y in [('train', Xtr, ytr), ('val', Xva, yva), ('test', Xte, yte)]:
    print(name, X.shape, dict(zip(NAMES, np.bincount(y))))
```

**Output:** (expected train `(400, 128, 128)`, val and test `(120, 128, 128)`, 100/30/30 per class)
```
PASTE OUTPUT HERE
```

```python
%matplotlib inline
fig, ax = plt.subplots(4, 4, figsize=(8, 8))
for c in range(4):
    for j, i in enumerate(np.where(ytr == c)[0][:4]):
        ax[c, j].imshow(Xtr[i], cmap='gray')
        ax[c, j].axis('off')
    ax[c, 0].set_title(NAMES[c], fontsize=9, loc='left')
plt.tight_layout()
plt.show()
```

**Output:** 4 example images per class
![q1_grid](images/q1_grid.png)

---

## Shared helper functions

```python
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def train_best_C(Xtr_f, ytr, Xva_f, yva):
    best_acc, best_model = -1, None
    for C in [0.1, 1, 10]:
        m = make_pipeline(StandardScaler(), LinearSVC(C=C, random_state=SEED, max_iter=20000))
        m.fit(Xtr_f, ytr)
        acc = accuracy_score(yva, m.predict(Xva_f))
        print("C =", C, "val acc =", round(acc, 4))
        if acc > best_acc:
            best_acc, best_model = acc, m
    return best_model

def report(model, Xf, y, title):
    p = model.predict(Xf)
    print(title, "accuracy:", round(accuracy_score(y, p), 4),
          "macro-F1:", round(f1_score(y, p, average='macro'), 4))
    cm = confusion_matrix(y, p)
    print(cm)
    ConfusionMatrixDisplay(cm, display_labels=NAMES).plot(xticks_rotation=45)
    plt.show()
    return cm
```

---

## Q2. HOG + SVM (6 marks)

### Q2.1 HOG features and feature length

```python
from skimage.feature import hog
from skimage import exposure

def hog_feat(img):
    return hog(img, orientations=9, pixels_per_cell=(HOG_CELL, HOG_CELL),
               cells_per_block=(2, 2), block_norm='L2-Hys')

H_tr = np.array([hog_feat(i) for i in Xtr])
H_va = np.array([hog_feat(i) for i in Xva])
print("HOG length:", H_tr.shape[1])
```

**Output:** (expected 8100)
```
PASTE OUTPUT HERE
```

**Hand calculation:**

- Image is 128 x 128 and the cell is 8 x 8, so 128 / 8 = 16 cells per side.
- Blocks are 2 x 2 cells moving 1 cell at a time, so (16 - 1) x (16 - 1) = 15 x 15 = 225 blocks.
- Each block has 2 x 2 cells x 9 orientations = 36 values.
- Feature length = 225 x 36 = **8100**, which matches the printed length.

### Q2.2 HOG pictures (one image per class)

```python
fig, ax = plt.subplots(4, 2, figsize=(5, 10))
for c in range(4):
    img = Xtr[np.where(ytr == c)[0][0]]
    _, pic = hog(img, orientations=9, pixels_per_cell=(HOG_CELL, HOG_CELL),
                 cells_per_block=(2, 2), visualize=True)
    ax[c, 0].imshow(img, cmap='gray'); ax[c, 0].set_title(NAMES[c]); ax[c, 0].axis('off')
    ax[c, 1].imshow(exposure.rescale_intensity(pic, in_range=(0, 10)), cmap='gray'); ax[c, 1].axis('off')
plt.show()
```

![q2_hog](images/q2_hog.png)

### Q2.3 Train SVM (C chosen on validation)

```python
hog_model = train_best_C(H_tr, ytr, H_va, yva)
cm_hog = report(hog_model, H_va, yva, "HOG val")
```

**Output:**
```
PASTE OUTPUT HERE
```

| Metric (validation) | Value |
|---|---|
| Chosen C | ___ |
| Accuracy | ___ |
| Macro-F1 | ___ |

![q2_cm](images/q2_cm.png)

---

## Q3. SIFT + SVM (7 marks)

### Q3.1 Normal SIFT keypoints

```python
sift = cv2.SIFT_create()
counts = []
for img in Xtr:
    kp, des = sift.detectAndCompute(img, None)
    counts.append(len(kp))
counts = np.array(counts)
print("avg:", counts.mean(), "min:", counts.min(), "max:", counts.max(),
      "zero-keypoint images:", (counts == 0).sum())

kp, _ = sift.detectAndCompute(Xtr[0], None)
plt.imshow(cv2.drawKeypoints(Xtr[0], kp, None, flags=4)); plt.axis('off'); plt.show()
```

**Output:**
```
PASTE OUTPUT HERE
```

| Statistic | Value |
|---|---|
| Average keypoints per image | ___ |
| Minimum | ___ |
| Maximum | ___ |
| Images with zero keypoints | ___ |

![q3_kp](images/q3_keypoints.png)

**Why a different number of keypoints is a problem for an SVM:** An SVM needs every image to give a feature vector of the same length, but normal SIFT returns a different number of 128-d descriptors per image, so the vectors cannot be stacked into one matrix.

### Q3.2 Fixed grid of keypoints

```python
centres = (np.arange(GRID) + 0.5) * (128 / GRID)    # 16, 48, 80, 112

def sift_grid_feat(img):
    kps = [cv2.KeyPoint(float(x), float(y), 32) for y in centres for x in centres]
    kps, des = sift.compute(img, kps)
    return des.flatten()

S_tr = np.array([sift_grid_feat(i) for i in Xtr])
S_va = np.array([sift_grid_feat(i) for i in Xva])
print("SIFT length:", S_tr.shape[1])
```

**Output:** (expected 2048)
```
PASTE OUTPUT HERE
```

**Hand calculation:** 4 x 4 grid = 16 grid points. Each SIFT descriptor has 128 values. Feature length = 16 x 128 = **2048**, which matches the printed length.

### Q3.3 Train SVM (C chosen on validation)

```python
sift_model = train_best_C(S_tr, ytr, S_va, yva)
cm_sift = report(sift_model, S_va, yva, "SIFT val")
```

**Output:**
```
PASTE OUTPUT HERE
```

| Metric (validation) | Value |
|---|---|
| Chosen C | ___ |
| Accuracy | ___ |
| Macro-F1 | ___ |

![q3_cm](images/q3_cm.png)

---

## Q4. Robustness (6 marks)

### Q4.1 Prediction (written BEFORE running the test)

> Write your own answer here before running the next cell.

- **Rotation:** I think ______ (HOG / SIFT) will lose more accuracy, because ______.
- **Noise:** I think ______ (HOG / SIFT) will lose more accuracy, because ______.

### Q4.2 Test on clean, rotated (20 degrees) and noisy (sigma = 0.05)

```python
def rotate(img):
    M = cv2.getRotationMatrix2D((64, 64), 20, 1.0)
    return cv2.warpAffine(img, M, (128, 128))

noise_rng = np.random.RandomState(SEED)
def add_noise(img):
    x = img / 255.0
    x = np.clip(x + noise_rng.normal(0, 0.05, x.shape), 0, 1)
    return (x * 255).astype(np.uint8)

cases = {'clean': Xte,
         'rotated': np.array([rotate(i) for i in Xte]),
         'noisy': np.array([add_noise(i) for i in Xte])}

rows = []
for name, model, fe in [('HOG + SVM', hog_model, hog_feat), ('SIFT + SVM', sift_model, sift_grid_feat)]:
    row = {'Model': name}
    for case, imgs in cases.items():
        F = np.array([fe(i) for i in imgs])
        row[case] = round(accuracy_score(yte, model.predict(F)), 4)
    rows.append(row)
display(pd.DataFrame(rows))
```

**Output:**
```
PASTE OUTPUT HERE
```

| Model | Clean accuracy | Rotated accuracy | Noisy accuracy |
|---|---|---|---|
| HOG + SVM | ___ | ___ | ___ |
| SIFT + SVM | ___ | ___ | ___ |

### Q4.3 Was my prediction correct? (three sentences, using the table)

1. My prediction was (correct / partly correct / wrong).
2. HOG went from ___ (clean) to ___ (rotated) and ___ (noisy); SIFT went from ___ to ___ and ___.
3. So ______ lost more under rotation and ______ lost more under noise.

---

## Q5. Short answers and AI log (3 marks)

### Q5.1 Short answers

**(a) Why does HOG use block normalisation?**
It rescales the gradient values inside each block, so the features are less affected by changes in lighting and contrast between images. This makes the descriptor more stable across images with different brightness.

**(b) Why is it wrong to choose C using the test set?**
The test set is meant to measure performance on unseen data. If C is chosen using it, the model has indirectly seen the test data, so the reported test accuracy is too optimistic. C must be chosen on the validation set.

### Q5.2 Most confused classes

```python
def top_confusion(cm, title):
    c = cm.copy().astype(float); np.fill_diagonal(c, 0)
    i, j = np.unravel_index(c.argmax(), c.shape)
    print(f"{title}: true {NAMES[i]} predicted as {NAMES[j]} ({int(c[i, j])} times)")

top_confusion(cm_hog, "HOG")
top_confusion(cm_sift, "SIFT")
```

**Output:**
```
PASTE OUTPUT HERE
```

- HOG confused ______ as ______ most.
- SIFT confused ______ as ______ most.
- **Why grayscale makes this harder:** the blood-cell types differ partly in stain colour. Grayscale removes colour, so only shape and brightness remain, and classes with similar shape look alike.

### Q5.3 AI log

| Prompt (one line) | What I changed | How I checked the result |
|---|---|---|
| ______ | ______ | ______ |
| ______ | ______ | ______ |
| ______ | ______ | ______ |

---

## Submission checklist

- [ ] Notebook named `RollNo_Name_LabMid.ipynb`, run with all outputs visible
- [ ] This MD file (best made with `jupyter nbconvert --to markdown RollNo_Name_LabMid.ipynb`) pushed to GitHub with its images folder
- [ ] PDF copy saved
- [ ] Printed shapes match: train (400,128,128), val/test (120,128,128); HOG length 8100; SIFT length 2048
- [ ] Test set used only in Q4
