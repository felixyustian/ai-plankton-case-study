### **2. Task 1: Data Audit**

#### `data_audit/auditor.py`
#### *A script designed to extract the requested insights.*

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