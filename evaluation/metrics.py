import numpy as np
import pandas as pd

def calculate_counting_metrics(gt_counts, pred_counts):
    """
    gt_counts: dict {image_id: {class_id: count}}
    pred_counts: dict {image_id: {class_id: count}}
    """
    
    errors = []
    
    for img_id, gt_class_counts in gt_counts.items():
        pred_class_counts = pred_counts.get(img_id, {})
        
        # Total count per image (Density Analysis)
        total_gt = sum(gt_class_counts.values())
        total_pred = sum(pred_class_counts.values())
        
        # Error calculation
        diff = total_pred - total_gt
        abs_diff = abs(diff)
        
        # Density Category
        if total_gt < 10: density = 'Sparse'
        elif total_gt <= 50: density = 'Medium'
        else: density = 'Dense'
        
        errors.append({
            'image_id': img_id,
            'total_gt': total_gt,
            'diff': diff, # Signed Error
            'abs_diff': abs_diff, # Absolute Error
            'density': density
        })
        
    df = pd.DataFrame(errors)
    
    results = {
        'MAE': df['abs_diff'].mean(),
        'Signed_Mean_Error': df['diff'].mean(), # Positive = Overcounting
        'MAE_Sparse': df[df['density']=='Sparse']['abs_diff'].mean(),
        'MAE_Dense': df[df['density']=='Dense']['abs_diff'].mean()
    }
    
    return results, df

if __name__ == "__main__":
    # Dummy data for demonstration
    print("Running Counting Metrics on dummy data...")
    dummy_gt = {'img1': {0: 5, 1: 2}, 'img2': {0: 60}}
    dummy_pred = {'img1': {0: 5, 1: 1}, 'img2': {0: 45}} 
    metrics, _ = calculate_counting_metrics(dummy_gt, dummy_pred)
    print(metrics)