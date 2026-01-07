# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import functional as F
try:
    from torchvision import datapoints
except ImportError:
    from torchvision import tv_tensors as datapoints

import os
import numpy as np
import cv2
from PIL import Image
import copy
import xml.etree.ElementTree as ET

class_to_id = {
    "Background": 0,
    "Bottle": 1,
    "Can": 2,
    "Chain": 3,
    "Drink-carton": 4,
    "Hook": 5,
    "Propeller": 6,
    "Shampoo-bottle": 7,
    "Standing-bottle": 8,
    "Tire": 9,
    "Valve": 10,
    "Wall": 11
}

def load_coco_box(xml_path):
    boxes = []
    if not os.path.exists(xml_path):
        # Return empty list if box file doesn't exist (robustness)
        return boxes
        
    tree = ET.ElementTree(file=xml_path)
    obj_list = tree.findall('object')
    for obj in obj_list:
        name = obj.find('name').text
        box = obj.find('bndbox')
        x = int(box.find('x').text)
        y = int(box.find('y').text)
        w = int(box.find('w').text)
        h = int(box.find('h').text)
        boxes.append([x, y, w, h, name])
    return boxes

def imread_cn(path):
    try:
        # Check if file exists first to give better error messages
        if not os.path.exists(path):
            return None
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

class DebrisDataset(Dataset):
    def __init__(self, root_path, image_list, input_size=1000, use_augment=False):
        # Only store PATHS, not actual images
        self.image_paths = []
        self.mask_paths = []
        self.box_paths = []
        
        self.use_augment = use_augment
        self.input_size = input_size
        self.normalize = transforms.Normalize(mean=[0.1701], std=[0.1848])

        # Transforms configuration
        if use_augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(
                    size=input_size,
                    scale=(0.6, 1.0),
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.ColorJitter(contrast=0.2, brightness=0.2),
            ])
        else:
            self.transform = transforms.Identity()

        image_dir = os.path.join(root_path, 'Images')
        mask_dir = os.path.join(root_path, 'Masks')
        box_dir = os.path.join(root_path, 'BoxAnnotations')

        with open(image_list, 'r') as f:
            lines = f.readlines()

        # Store paths only
        for line in lines:
            line = line.strip()
            if not line: continue
            
            self.image_paths.append(os.path.join(image_dir, line + '.png'))
            self.mask_paths.append(os.path.join(mask_dir, line + '.png'))
            self.box_paths.append(os.path.join(box_dir, line + '.xml'))

        print(f"Index loaded: {len(self.image_paths)} samples found in {image_list}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. LOAD IMAGE
        img_path = self.image_paths[idx]
        image = imread_cn(img_path)
        
        if image is None:
            raise ValueError(f"!!! CRASH DETECTED !!! Could not load image at: {img_path}")

        # --- CLAHE IMPLEMENTATION START ---
        # Domain Adaptation: Sonar images are often low-contrast with speckle noise.
        # We apply CLAHE to enhance local contrast without amplifying noise globally.
        
        # 1. Ensure image is single-channel grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply CLAHE (ClipLimit=2.0 is standard for Sonar)
        # clipLimit: Threshold for contrast limiting (prevents noise explosion)
        # tileGridSize: Size of area for local histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply(image)
        
        # 3. Convert back to RGB (SAM requires 3 channels)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # --- CLAHE IMPLEMENTATION END ---
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # 2. LOAD MASK
        mask_path = self.mask_paths[idx]
        mask = imread_cn(mask_path)
        if mask is None:
             raise ValueError(f"!!! CRASH DETECTED !!! Could not load mask at: {mask_path}")

        # 3. LOAD BOXES
        box_path = self.box_paths[idx]
        boxes_ori = load_coco_box(box_path)

        # 4. PRE-PROCESS (Padding & Resizing)
        # This logic was moved from __init__ to here
        h, w, _ = image.shape
        side = max(h, w)
        pad_u = pad_b = pad_l = pad_r = 0

        if h > w:
            pad = h - w
            pad_l = pad // 2
            pad_r = pad - pad_l
        else:
            pad = w - h
            pad_u = pad // 2
            pad_b = pad - pad_u

        # Pad and Resize Image
        image = cv2.copyMakeBorder(image, pad_u, pad_b, pad_l, pad_r,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
        image = cv2.resize(image, (self.input_size, self.input_size))

        # Pad and Resize Mask (Create 12 channel mask on the fly)
        mask_list = []
        for i in range(12):
            m = (mask == i).astype("uint8")
            m = cv2.copyMakeBorder(m, pad_u, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT)
            m = cv2.resize(m, (self.input_size, self.input_size), cv2.INTER_NEAREST)
            mask_list.append(m)

        # Prepare Boxes
        scale = self.input_size / side
        boxes = []
        labels = []
        box_coords = []
        
        for x, y, w, h, cls in boxes_ori:
            x1 = int((x + pad_l) * scale)
            y1 = int((y + pad_u) * scale)
            x2 = int((x + w + pad_l) * scale)
            y2 = int((y + h + pad_u) * scale)
            
            boxes.append([x1, y1, x2, y2, class_to_id[cls]])
            box_coords.append([x1, y1, x2, y2])
            labels.append(class_to_id[cls])

        # 5. CONVERT TO TORCHVISION TENSORS
        pil_image = Image.fromarray(image)
        tv_image = datapoints.Image(pil_image)

        # Merge masks for transformation
        temp = np.zeros_like(mask_list[0])
        for i, m in enumerate(mask_list):
            temp += i * m
        tv_mask = datapoints.Mask(Image.fromarray(temp))

        # Handle empty boxes case (some images might have no objects)
        if len(box_coords) == 0:
            # Create a dummy box to satisfy transform pipeline, will filter later if needed
            # Or handle specifically based on model requirements. 
            # For now, assuming dataset has boxes. If empty, create 0,0,0,0
            box_coords = [[0, 0, 1, 1]] 
            labels = [0] 

        # tv_bboxes = datapoints.BoundingBox(
        #     box_coords,
        #     format=datapoints.BoundingBoxFormat.XYXY,
        #     spatial_size=F.get_spatial_size(tv_image)
        # )

        # FIX: Check if using new API (BoundingBoxes) or old API (BoundingBox)
        if hasattr(datapoints, "BoundingBoxes"):
            # New Torchvision (v0.16+)
            tv_bboxes = datapoints.BoundingBoxes(
                box_coords,
                format=datapoints.BoundingBoxFormat.XYXY,
                canvas_size=tuple(tv_image.shape[-2:]) # <--- CHANGED HERE
            )
        else:
            # Old Torchvision (v0.15)
            tv_bboxes = datapoints.BoundingBox(
                box_coords,
                format=datapoints.BoundingBoxFormat.XYXY,
                spatial_size=tuple(tv_image.shape[-2:]) # <--- CHANGED HERE TOO (for safety)
            )

        # 6. APPLY TRANSFORMS
        # Note: We consolidated transform_train and transform_test into self.transform
        tv_image, tv_bboxes, tv_mask, labels = self.transform(tv_image, tv_bboxes, tv_mask, labels)

        # 7. FINALIZE OUTPUTS
        # final_image = self.normalize(F.to_image(tv_image))

        # FIX: Convert to Float and Scale to [0, 1] before normalizing
        img_tensor = F.to_image(tv_image)

        # Ensure it is float32 and scaled to 0-1 range
        if img_tensor.dtype == torch.uint8:
            img_tensor = img_tensor.to(dtype=torch.float32) / 255.0

        final_image = self.normalize(img_tensor)

        # FIX: Squeeze to remove the extra channel dimension
        mask_np = np.array(F.to_image(tv_mask)).squeeze()
        masks_out = []
        for i in range(12):
            masks_out.append((mask_np == i).astype("float32"))
        masks_out = torch.tensor(np.stack(masks_out))

        # --- NEW LOGIC STARTS HERE ---
        final_boxes = []
        final_labels = []  # <--- Define the missing variable!
        
        # Iterate over boxes and labels together
        for box, label in zip(tv_bboxes, labels):
            x1, y1, x2, y2 = map(int, box)
            
            # Filter invalid boxes (width or height <= 0)
            if x2 <= x1 or y2 <= y1: continue 
            
            # Append valid box and its corresponding label
            final_boxes.append([x1, y1, x2, y2])
            final_labels.append(label)
        
        # --- NEW LOGIC ENDS HERE ---

        # Now 'final_labels' exists and can be returned
        return final_image, masks_out, {"boxes": final_boxes, "labels": final_labels}

def collate_fn_seq(batch):
    images = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    targets = [b[2] for b in batch]
    return images, masks, targets

def collate_fn_seq_box_seg_pair(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets