# Plankton Detection & Counting: Technical Case Study

## Overview
This repository contains the analysis, evaluation logic, and improvement prototypes for the Hadl.ai plankton counting challenge.

**Core Philosophy:** In aquaculture monitoring, *count accuracy* often outweighs pure bounding box precision. A model that hallucinates plankton in clear water is worse than one that slightly misaligns a box on an existing one. This solution prioritizes **Density-Awareness** and **Data-Centric** corrections.

## Approach Summary
1.  **Data Audit:** utilized embedding-space visualization (t-SNE) to detect intra-class variance and "Copy-Paste" simulation to estimate occlusion limits.
2.  **Baseline Model:** YOLOv8-Large trained on 640sz (simulated).
3.  **Key Finding:** The model fails significantly in "clumping" scenarios (high overlap), leading to under-counting due to NMS (Non-Maximum Suppression) suppression.

## Repository Map
* `data_audit/`: Class imbalance analysis and embedding topology.
* `evaluation/`: Custom "Counting Metrics" (MAE, Signed Error) and density-based error breakdowns.
* `improvement/`: Proposal for "Density Map Regression" head and **Prototype Implementation of Copy-Paste Augmentation**.

## Instructions
To run the evaluation metrics:
```bash
python evaluation/metrics.py --preds predictions.json --gt ground_truth.json
