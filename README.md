# Plankton Detection & Counting: Technical Case Study

## Overview
This repository contains the analysis, evaluation logic, and improvement prototypes for the Hadl.ai plankton counting challenge.

### ✨ The "Magic": A Biological & Data-Centric Manifesto
While standard CV approaches focus on optimizing IoU, my "magic" lies in applying **biological constraints** to computer vision. Plankton are not random rigid objects; they are fluid-dynamic entities that cluster.
* **The Insight:** A model that hallucinates plankton in clear water is functionally worse than one that misaligns a box on a real organism.
* **The Approach:** Instead of blindly tuning hyperparameters, I prioritize **Density-Awareness** (treating clusters differently than isolates) and **Data Geometry** (fixing the 1:2500 imbalance). I treat the annotation pipeline as a software product, not just a static input.

## Key Outcomes
* **Data Audit:** Identified a severe **1:2500 class imbalance** (Chlorella vs Vorticella) and systematic "smearing" annotation errors caused by motion blur.
* **Evaluation:** Proved that **NMS (Non-Maximum Suppression)** is the primary bottleneck in high-density scenes, leading to systematic under-counting.
* **Improvement:** Implemented a **Copy-Paste Augmentation** prototype to synthetically upsample rare classes.

## Repository Map
```text
hadl-plankton-case-study/
├── assets/                  # Generated visual evidence (Distributions, Failures, Variance)
├── data_audit/              # Geometric analysis & class distribution scripts
│   └── README.md            # Deep dive into Imbalance & variance
|   └── analyze_coco.py      # Run the geometric analysis & class distribution scripts
├── evaluation/              # Custom metrics & Failure analysis
│   └── README.md            # Analysis of density bias & counting errors
|   └── density_metrics.py   # Run the custom counting metrics (MAE, Signed Error) against the ground truth
└── improvement/             # Prototyping solutions
    ├── README.md            # Improvement proposals & prioritization Matrix
    └── coco_copy_paste.py   # PROTOTYPE: Rare class injection strategy
```

## Quick Start
To reproduce the visual analysis and audit the dataset metadata (generates images in /assets):

```bash
python generate_all_assets.py
```

To run the custom counting metrics (MAE, Signed Error) against the ground truth:
```bash
python evaluation/density_metrics.py --preds predictions.json --gt _annotations.coco.json
```

## Approach Summary
### 1. Data-Centric Audit
We moved beyond simple class counting to analyze Bounding Box Geometry.

Findings: The dataset is heavily skewed towards Chlorella sp (2,500+ instances), while tail classes like Vorticella have single-digit instances.

Quality Control: We detected 138 suspect annotations characterized by extreme aspect ratios (>10:1). These likely represent motion blur artifacts ("smears") rather than valid organisms, which confuses the model.

### 2. Density-Aware Evaluation
Standard mAP is insufficient for this business case because it hides failures in crowded scenes. We introduced Density Binning:

Sparse (<10 objects): High accuracy.

Dense (>50 objects): Accuracy degrades significantly due to overlapping bounding boxes being suppressed by the detector's NMS.

### 3. Improvement Strategy & Prioritization
#### 3.1 Proposals & Trade-offs
We evaluated solutions based on their impact on the specific "Clumping" and "Imbalance" problems.

| Proposal | Target Problem | Expected Impact | Expected Impact |
| :--- | :--- | :--- | :--- |
| Expected Impact | Class Imbalance | High recall for rare species | Artifact Risk: Pasting plankton on "noisy" backgrounds (bubbles) may create unrealistic training samples that confuse the model |
| Density Map Regression | Occlusion/Clumping | Solves NMS failure completely | Complexity: Requires changing model architecture (YOLO -> U-Net) and potentially re-labeling data as points (dots) instead of boxes |
| Soft-NMS | Occlusion | Moderate improvement in dense clusters | Inference Latency: Increases post-processing time linearly with the number of detections |

#### 3.2 Prioritization (2-Week Sprint)
##### Priority 1: Copy-Paste Augmentation (Implemented)

Why: It is a low-code, high-impact intervention that directly solves the 1:2500 imbalance without requiring architectural changes. It leverages existing data to "create" new data.

##### Priority 2: Soft-NMS Integration

Why: A quick algorithmic tweak to reduce under-counting in clusters.

##### DEFERRED: Density Map Regression

Rationale: While theoretically superior for counting, the engineering cost is too high for a 2-week sprint. It requires a fundamental shift in the pipeline (Point Supervision) and retraining from scratch. I would only pursue this if Soft-NMS fails to meet accuracy KPIs.

#### 3.3 Prototype: Copy-Paste Augmentation
I implemented the coco_copy_paste.py script (see improvement/).

What it does: Extracts rare classes from source images and seamlessly clones them into training images using Poisson Blending.

Honest Assessment: The prototype works well on clear backgrounds, but struggles when pasting onto complex textures (e.g., debris), occasionally creating "halo" artifacts. A production version would require a segmentation-based mask check.

---

## *Submitted by Felix Yustian for Hadl.ai Senior Computer Vision Scientist Role*
