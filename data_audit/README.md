# Task 1: Data Audit & Analysis

## 1.1 Class Distribution Analysis
We analyzed the dataset and identified a severe long-tail imbalance.
* **Dominant Class:** *Chlorella sp*
* **Imbalance Ratio:** >1:2000 (standard Cross-Entropy loss will likely fail).
* **Strategy:** Recommended Mosaic Augmentation and Copy-Paste for rare classes.

**Visual Evidence:**
![Class Distribution](../assets/class_distribution.png)

## 1.2 Intra-Class Variance
We examined the geometric variance within classes.
* **Observation:** *Oscillatoria sp* exhibits significant morphological variation (filamentous vs. fragmented), suggesting the model needs deformable convolutions or high rotation augmentation.

**Visual Evidence:**
![Intra-Class Variance](../assets/intra_class_variance.png)

## 1.3 Inter-Class Similarity
We analyzed confusion risks between geometrically similar classes.
* **Observation:** *Chlorella* and *Pyramimonas* have nearly identical bounding box aspect ratios (1:1), making them indistinguishable without high-resolution texture features.

**Visual Evidence:**
![Inter-Class Confusion](../assets/inter_class_confusion.png)

## 1.4 Annotation Quality
We audited the ground truth and identified **138 potential errors**, primarily "Smearing" (Motion Blur) labeled as distinct objects.
* **Recommendation:** Apply 'Confident Learning' to automatically flag and clean these labels.

**Visual Evidence:**
![Bad Annotations](../assets/bad_annotations.png)