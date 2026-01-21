# Plankton Detection & Counting: Technical Case Study

## Overview
This repository contains the analysis, evaluation logic, and improvement prototypes for the Hadl.ai plankton counting challenge.

**Core Philosophy:** In aquaculture monitoring, *count accuracy* often outweighs pure bounding box precision. A model that hallucinates plankton in clear water is worse than one that slightly misaligns a box on an existing one. This solution prioritizes **Density-Awareness** and **Data-Centric** corrections over simple hyperparameter tuning.

## Key Outcomes
* **Data Audit:** Identified a severe **1:2500 class imbalance** and systematic "smearing" annotation errors caused by motion blur.
* **Evaluation:** Proved that **NMS (Non-Maximum Suppression)** is the primary bottleneck in high-density scenes, leading to systematic under-counting.
* **Improvement:** Implemented a **Copy-Paste Augmentation** prototype to synthetically upsample rare classes and simulate occlusion.

## Repository Map
```text
hadl-plankton-case-study/
├── assets/                  # Generated visual evidence (Distributions, Failures, Variance)
├── data_audit/              # Geometric analysis & Class distribution scripts
│   └── README.md            # Deep dive into Imbalance & Variance
├── evaluation/              # Custom metrics & Failure analysis
│   └── README.md            # Analysis of Density Bias & Counting Errors
└── improvement/             # Prototyping solutions
    ├── README.md            # Improvement Proposals & Prioritization Matrix
    └── coco_copy_paste.py   # PROTOTYPE: Rare Class Injection strategy
```

## Quick Start
To reproduce the visual analysis and audit the dataset metadata:

## Generates the 5 required visual assets in /assets folder
```bash
python generate_all_assets.py
```

## To run the custom counting metrics (MAE, Signed Error):
```bash
python evaluation/metrics.py --preds predictions.json --gt _annotations.coco.json
```

## Approach Summary
## 1. Data-Centric Audit
We moved beyond simple class counting to analyze Bounding Box Geometry.

Findings: The dataset is dominated by Chlorella sp, while classes like Vorticella have single-digit instances.

Quality Control: We detected 138 suspect annotations (extreme aspect ratios) representing motion blur artifacts rather than valid organisms.

## 2. Density-Aware Evaluation
Standard mAP is insufficient for this business case. We introduced Density Binning:

Sparse (<10 objects): High accuracy.

Dense (>50 objects): Accuracy degrades significantly due to overlapping bounding boxes.

## 3. Proposed Solution
Immediate Term: Copy-Paste Augmentation (Implemented in improvement/) to fix the long-tail distribution.

## Long Term Solution: 
Switch detection head to Density Map Regression to bypass NMS limitations in crowded tanks.

# *Submitted by Felix Yustian for Hadl.ai Senior Computer Vision Scientist Role*
