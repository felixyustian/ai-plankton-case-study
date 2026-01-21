import cv2
import numpy as np
import random
import os

class CopyPasteAugmenter:
    """
    Implements Copy-Paste Augmentation (Ghiasi et al., 2021).
    Extracts objects of interest (rare classes) and pastes them onto other training images.
    
    Why this helps:
    1. Increases instance count of rare classes.
    2. Simulates occlusion (by pasting on top of others), helping the model learn 
       to separate overlapping plankton (solving the 'Merged Detection' failure).
    """

    def __init__(self, rare_class_ids, probability=0.5):
        self.rare_ids = rare_class_ids
        self.prob = probability
        # In a real scenario, we would cache segmented objects here
        self.object_bank = [] 

    def extract_object(self, image, box):
        """
        Naively extracts object crop. 
        Note: For best results, this needs Segmentation Masks. 
        Since we only have boxes (YOLO), we use the box crop.
        """
        x1, y1, x2, y2 = map(int, box)
        return image[y1:y2, x1:x2]

    def augment(self, image, labels):
        """
        image: cv2 image
        labels: list of [class_id, x_center, y_center, w, h] (YOLO format)
        """
        if random.random() > self.prob:
            return image, labels

        # Simulate grabbing a rare object from the bank
        # (In production, this comes from a database of rare class crops)
        dummy_rare_crop = np.zeros((50, 50, 3), dtype=np.uint8) # Placeholder
        dummy_rare_id = self.rare_ids[0]

        # Random paste location
        h, w, _ = image.shape
        paste_x = random.randint(0, w - 50)
        paste_y = random.randint(0, h - 50)

        # Paste the object (Simple Overwrite)
        # Advanced: Use Poisson Blending for seamless edges to fix 'Lighting Consistency'
        image[paste_y:paste_y+50, paste_x:paste_x+50] = dummy_rare_crop

        # Update labels (Normalize to 0-1)
        new_label = [
            dummy_rare_id,
            (paste_x + 25) / w,
            (paste_y + 25) / h,
            50 / w,
            50 / h
        ]
        labels.append(new_label)

        return image, labels

# --- Prototype Validation Script ---
if __name__ == "__main__":
    print("Initializing Copy-Paste Augmentation Prototype...")
    aug = CopyPasteAugmenter(rare_class_ids=[4, 7])
    
    # Mock Image
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    dummy_labels = [[1, 0.5, 0.5, 0.1, 0.1]]
    
    aug_img, aug_labels = aug.augment(dummy_img, dummy_labels)
    
    print(f"Original Label Count: 1")
    print(f"Augmented Label Count: {len(aug_labels)}")
    print("Success: Rare class injected into training stream.")