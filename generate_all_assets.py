import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
import numpy as np

# --- CONFIGURATION ---
JSON_FILE = '_annotations.coco.json'
OUTPUT_DIR = 'assets'
# ---------------------

def setup():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    with open(JSON_FILE, 'r') as f:
        return json.load(f)

def generate_task_1_1_distribution(coco):
    """Task 1.1: Class Distribution Analysis"""
    print("Generating 1.1 Class Distribution...")
    cat_names = {c['id']: c['name'] for c in coco['categories']}
    counts = {}
    for ann in coco['annotations']:
        name = cat_names[ann['category_id']]
        counts[name] = counts.get(name, 0) + 1
    
    df = pd.DataFrame(list(counts.items()), columns=['Class', 'Count']).sort_values('Count', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Count', y='Class', palette='viridis')
    plt.xscale('log') # Crucial for the 1:2500 imbalance
    plt.title('Task 1.1: Class Distribution (Log Scale)\nHighlighting Severe Imbalance')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/class_distribution.png')
    plt.close()

def generate_task_1_2_variance(coco):
    """Task 1.2: Intra-class Variance (Geometric Overlay)"""
    print("Generating 1.2 Intra-class Variance...")
    # Find class with variable shapes (Oscillatoria sp is filamentous)
    target_id = 0 # Oscillatoria
    anns = [a for a in coco['annotations'] if a['category_id'] == target_id]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"Task 1.2: Intra-Class Variance (Oscillatoria sp)\nOverlay of {min(50, len(anns))} instances")
    
    for i, ann in enumerate(anns[:50]):
        w, h = ann['bbox'][2], ann['bbox'][3]
        # Center at 0,0 to compare shapes
        rect = patches.Rectangle((-w/2, -h/2), w, h, linewidth=1, edgecolor='blue', facecolor='none', alpha=0.3)
        ax.add_patch(rect)
        
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_aspect('equal')
    plt.savefig(f'{OUTPUT_DIR}/intra_class_variance.png')
    plt.close()

def generate_task_1_3_similarity(coco):
    """Task 1.3: Inter-class Similarity (Confusion Scatter)"""
    print("Generating 1.3 Inter-class Similarity...")
    # Compare Chlorella (1) vs Pyramimonas (3) - similar small shapes
    data = []
    for ann in coco['annotations']:
        if ann['category_id'] in [1, 3]:
            data.append({
                'Class': 'Chlorella' if ann['category_id'] == 1 else 'Pyramimonas',
                'Width': ann['bbox'][2],
                'Height': ann['bbox'][3]
            })
    
    if not data: return
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Width', y='Height', hue='Class', alpha=0.5)
    plt.title('Task 1.3: Inter-Class Similarity\nGeometric Overlap = Confusion Risk')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.savefig(f'{OUTPUT_DIR}/inter_class_confusion.png')
    plt.close()

def generate_task_1_4_bad_annotations(coco):
    """Task 1.4: Bad Annotations (The 'Top 5 Weirdest' Boxes)"""
    print("Generating 1.4 Bad Annotations...")
    # Find annotations with extreme aspect ratios (Smears)
    anns = coco['annotations']
    anns.sort(key=lambda x: x['bbox'][2]/x['bbox'][3] if x['bbox'][3]>0 else 0, reverse=True)
    top_5 = anns[:5]
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    plt.suptitle("Task 1.4: Detected Annotation Errors (Extreme Aspect Ratios)")
    
    for i, ann in enumerate(top_5):
        w, h = ann['bbox'][2], ann['bbox'][3]
        ax = axes[i]
        # Draw the box on a blank canvas
        rect = patches.Rectangle((10, 50-h/2), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.set_xlim(0, max(w+20, 100))
        ax.set_ylim(0, 100)
        ax.set_title(f"ID: {ann['id']}\nRatio: {w/h:.1f}:1")
        ax.axis('off')
        
    plt.savefig(f'{OUTPUT_DIR}/bad_annotations.png')
    plt.close()

def generate_task_2_2_failure_modes(coco):
    """Task 2.2: Failure Modes (Occlusion, Blur, Debris)"""
    print("Generating 2.2 Failure Modes...")
    # 1. Occlusion: Image 114 has overlap
    occ_anns = [a for a in coco['annotations'] if a['image_id'] == 114]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot A: Occlusion (Real Data from Image 114)
    ax = axes[0]
    ax.set_title("Failure 1: Occlusion (Merged)\n(Image ID 114)")
    ax.set_xlim(0, 640); ax.set_ylim(640, 0)
    for ann in occ_anns:
        x, y, w, h = ann['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
    # Plot B: Blur (Extreme Ratio)
    ax = axes[1]
    ax.set_title("Failure 2: Motion Blur\n(High Aspect Ratio)")
    rect = patches.Rectangle((50, 50), 200, 10, linewidth=2, edgecolor='orange', facecolor='none')
    ax.add_patch(rect)
    ax.text(100, 40, "Smear Artifact", color='orange')
    ax.set_xlim(0, 300); ax.set_ylim(100, 0)
    
    # Plot C: Debris (Tiny)
    ax = axes[2]
    ax.set_title("Failure 3: Debris\n(Noise < 10px)")
    rect = patches.Rectangle((50, 50), 5, 5, linewidth=2, edgecolor='purple', facecolor='none')
    ax.add_patch(rect)
    ax.text(60, 50, "Bubble/Dirt", color='purple')
    ax.set_xlim(0, 100); ax.set_ylim(100, 0)
    
    plt.savefig(f'{OUTPUT_DIR}/failure_modes.png')
    plt.close()

if __name__ == "__main__":
    coco = setup()
    generate_task_1_1_distribution(coco)
    generate_task_1_2_variance(coco)
    generate_task_1_3_similarity(coco)
    generate_task_1_4_bad_annotations(coco)
    generate_task_2_2_failure_modes(coco)
    print(f"Done! All 5 assets saved to /{OUTPUT_DIR}")