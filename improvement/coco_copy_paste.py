import json
import cv2
import numpy as np
import random
from pycocotools.coco import COCO

class COCOCopyPaste:
    def __init__(self, ann_file, img_dir, rare_class_ids):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.rare_ids = rare_class_ids
        
        # Cache rare annotations
        self.rare_anns = []
        for cat_id in self.rare_ids:
            img_ids = self.coco.getImgIds(catIds=[cat_id])
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_ids))
            self.rare_anns.extend(anns)
            
    def get_random_rare_crop(self):
        """Extracts a pixel-perfect crop of a rare plankton."""
        ann = random.choice(self.rare_anns)
        img_info = self.coco.loadImgs(ann['image_id'])[0]
        
        # Load source image
        path = f"{self.img_dir}/{img_info['file_name']}"
        img = cv2.imread(path)
        
        x, y, w, h = [int(v) for v in ann['bbox']]
        
        # Safety check for image bounds
        if w <= 0 or h <= 0: return None
        
        crop = img[y:y+h, x:x+w]
        return crop, ann['category_id']

    def augment(self, target_img_path):
        img = cv2.imread(target_img_path)
        h_img, w_img, _ = img.shape
        
        # Paste 3 rare objects
        for _ in range(3):
            crop, cat_id = self.get_random_rare_crop()
            if crop is None: continue
            
            ch, cw, _ = crop.shape
            
            # Random position
            tx = random.randint(0, w_img - cw)
            ty = random.randint(0, h_img - ch)
            
            # Seamless Clone (Poisson Blending) - Critical for water background!
            # We create a mask for the crop
            mask = 255 * np.ones(crop.shape, img.dtype)
            center = (tx + cw//2, ty + ch//2)
            
            try:
                img = cv2.seamlessClone(crop, img, mask, center, cv2.NORMAL_CLONE)
            except:
                # Fallback to simple paste if clone fails
                img[ty:ty+ch, tx:tx+cw] = crop
                
        return img

# Usage
# augmenter = COCOCopyPaste('../_annotations.coco.json', '../train_images', [15, 18])
# new_img = augmenter.augment('../train_images/sample_01.jpg')