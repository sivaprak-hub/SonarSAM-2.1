import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Reuse your existing mapping
from dataloader.data_loader import class_to_id 

class SyntheticCropDataset(Dataset):
    def __init__(self, crop_root, samples_per_epoch=2000, input_size=1024):
        self.crop_root = crop_root
        self.samples_per_epoch = samples_per_epoch # How many "fake" images to generate per epoch
        self.input_size = input_size
        self.normalize = transforms.Normalize(mean=[0.1701], std=[0.1848])
        
        # Load all crop paths
        self.crops = []
        for cls_name, cls_id in class_to_id.items():
            if cls_id == 0 or cls_name == "Background": continue
            
            cls_dir = os.path.join(crop_root, cls_name)
            if os.path.exists(cls_dir):
                files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith('.png')]
                for f in files:
                    self.crops.append((f, cls_id))
        
        print(f"Synthetic Loader: Found {len(self.crops)} unique crops. Generating {samples_per_epoch} samples/epoch.")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # 1. Pick a random crop
        crop_path, cls_id = self.crops[np.random.randint(len(self.crops))]
        
        # 2. Load Crop (Grayscale)
        crop = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
        if crop is None:
            # Safe fallback if file is broken
            return self.__getitem__(0)
            
        h_c, w_c = crop.shape
        
        # 3. Create Background (Simulated Sonar Noise)
        # Create a base gray background with some Gaussian noise
        bg = np.random.normal(40, 10, (self.input_size, self.input_size)).astype(np.uint8)
        
        # 4. Paste Crop at Random Location
        x = np.random.randint(0, self.input_size - w_c - 1)
        y = np.random.randint(0, self.input_size - h_c - 1)
        
        # Paste the crop
        bg[y:y+h_c, x:x+w_c] = crop
        
        # 5. Generate Mask (Auto-Threshold)
        # Assume object pixels are brighter than pure black (threshold > 10)
        # You might need to adjust this threshold based on your crops
        mask_full = np.zeros((self.input_size, self.input_size), dtype=np.uint8)
        object_mask = (crop > 20).astype(np.uint8) 
        mask_full[y:y+h_c, x:x+w_c] = object_mask * cls_id

        # 6. Prepare Outputs for SAM
        # Convert Image to 3-channel RGB Tensor
        img_rgb = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
        final_image = self.normalize(img_tensor)
        
        # Prepare Masks (Stack 12 channels)
        masks_out = []
        for i in range(12):
            masks_out.append((mask_full == i).astype("float32"))
        masks_out = torch.tensor(np.stack(masks_out))
        
        # Prepare Box
        # We know exactly where we pasted it
        box = [x, y, x+w_c, y+h_c]
        
        return final_image, masks_out, {"boxes": [box], "labels": [cls_id]}