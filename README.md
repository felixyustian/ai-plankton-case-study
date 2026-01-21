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

---

### **2. Task 1: Data Audit**

#### `data_audit/auditor.py`
*A script designed to extract the requested insights.*

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def calculate_imbalance_metrics(labels):
    """
    Calculates Gini Coefficient and Imbalance Ratio.
    Senior approach: Don't just count; measure the 'inequality' of the distribution.
    """
    counts = Counter(labels)
    df = pd.DataFrame.from_dict(counts, orient='index', columns=['count'])
    df = df.sort_values('count', ascending=False)
    
    # Imbalance Ratio
    max_c = df['count'].max()
    min_c = df['count'].min()
    ratio = max_c / min_c
    
    print(f"Severe Imbalance Detected: Ratio 1:{ratio:.2f}")
    return df

def audit_intra_class_variance(features, labels, class_names):
    """
    Uses t-SNE on ResNet50 extracted features to visualize variance.
    If a class cluster is spread out, it has high intra-class variance.
    """
    # (Pseudocode for the 'Magic' approach mentioned in the prompt)
    # 1. Extract embeddings for all crops.
    # 2. Run t-SNE.
    # 3. Calculate Euclidean distance variance within class clusters.
    pass

# Note: This script assumes access to the dataset directory.
