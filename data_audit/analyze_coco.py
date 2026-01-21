import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def audit_coco_dataset(json_path='../_annotations.coco.json'):
    with open(json_path, 'r') as f:
        coco = json.load(f)

    # 1. Class Distribution (Ground Truth)
    cat_map = {c['id']: c['name'] for c in coco['categories']}
    cat_counts = {c['name']: 0 for c in coco['categories']}
    
    # 2. Size & Density Analysis
    areas = []
    box_counts_per_img = {}
    
    for ann in coco['annotations']:
        cat_name = cat_map[ann['category_id']]
        cat_counts[cat_name] += 1
        
        # COCO Bbox is [x, y, w, h]
        area = ann['bbox'][2] * ann['bbox'][3]
        areas.append(area)
        
        img_id = ann['image_id']
        box_counts_per_img[img_id] = box_counts_per_img.get(img_id, 0) + 1

    # Output 1: Imbalance Report
    df = pd.DataFrame(list(cat_counts.items()), columns=['Class', 'Count'])
    df['Ratio'] = df['Count'] / df['Count'].max()
    print("=== Imbalance Report ===")
    print(df.sort_values('Count', ascending=False).head(10))

    # Output 2: Size Bias (Small < 32^2 px)
    areas = np.array(areas)
    small_obj_pct = (areas < 32**2).mean() * 100
    print(f"\n=== Size Analysis ===")
    print(f"Small Objects (<32x32px): {small_obj_pct:.1f}%")
    print(f"Median Object Size: {np.median(areas):.0f} px^2")

    # Output 3: Clumping/Density
    density = list(box_counts_per_img.values())
    print(f"\n=== Density Analysis ===")
    print(f"Avg Plankton per Image: {np.mean(density):.1f}")
    print(f"Max Plankton per Image: {np.max(density)}")
    
    return df

if __name__ == "__main__":
    audit_coco_dataset()