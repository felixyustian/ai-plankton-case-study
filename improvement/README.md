# 3.1 Improvement Proposals

| Challenge | Proposed Approach | Expected Impact | Tradeoffs | Complexity |
| :--- | :--- | :--- | :--- | :--- |
| **Class Imbalance** | **Copy-Paste Augmentation** | High increase in recall for rare classes. | Risk of creating unrealistic artifacts if not blended well. | Low (Data loader only) |
| **Counting Accuracy** | **Density Map Regression Head** | Solves the "Merged Detection" issue. Instead of boxes, we sum the pixel density. | Requires modifying model architecture (YOLO + U-Net hybrid). | High |
| **Small Objects** | **SAHI (Slicing Aided Hyper Inference)** | Recovers small objects by running inference on crops. | Increases inference time linearly with number of slices. | Medium |

# 3.2 Prioritization (2 Weeks Engineering Time)

1.  **Week 1 (Quick Wins):** Implement **Copy-Paste Augmentation** (Solves Imbalance) and **SAHI** (Solves Small Object Misses). These require no architectural changes, just pipeline tweaks.
2.  **Week 2 (Deep Fix):** Investigate **Soft-NMS** or **WBF (Weighted Box Fusion)** to reduce aggressive suppression of overlapping plankton.
3.  **Defer:** Density Map Regression. While mathematically superior for counting, it requires a full retraining and potentially new annotation styles (dot annotations).

# 3.3 Prototype Implementation
**Selected:** Copy-Paste Augmentation.
**Rationale:** We detected a 1:120 imbalance. No amount of model tuning fixes missing data. We must "create" data.