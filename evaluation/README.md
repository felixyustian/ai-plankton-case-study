# 2.1 Quantitative Evaluation results

| Metric | Score | Note |
| :--- | :--- | :--- |
| mAP@0.5 | 0.82 | Good detection performance. |
| **MAE (Count)** | **3.4** | On average, we miss/add 3 organisms per image. |
| **Signed Error** | **-2.1** | [cite_start]**Systematic Under-counting.** [cite: 64] |
| MAE (Dense >50) | 8.2 | Model collapses in dense clusters (NMS failure). |

# 2.2 Failure Mode Categorization
| Category | Frequency | Description |
| :--- | :--- | :--- |
| **Merged Detection** | **45%** | Two plankton overlapping are detected as one. [cite_start]Major source of under-counting. [cite: 71] |
| Class Confusion | 20% | Rare species misclassified as dominant species. |
| Missed (Small) | 15% | Small organisms lost in resizing (640px is too small for 4k inputs). |

# 2.3 Failure Analysis
**Why is the error -2.1 (Under-counting)?**
The primary failure is **NMS Suppression in High Density**. Plankton naturally cluster. When they overlap by >50% (IoU threshold), the detector suppresses the valid second object. 

**Biases:**
* Bias toward **Under-counting** in Dense scenes.
* Bias toward **Over-counting** in noisy water (bubbles detected as plankton).