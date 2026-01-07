import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import datetime
import numpy as np
import cv2
import shutil
from tqdm import tqdm
from torch.utils.data import DataLoader

# Imports
from model.model_proxy_SAM import SonarSAM 
from utils.config import Config_SAM
from utils.logger import Logger
from utils.utils import rand_seed
from dataloader.data_loader import DebrisDataset, collate_fn_seq, class_to_id
from model.loss_functions import dice_loss

# Reverse map for visualization (ID -> Name)
id_to_class = {v: k for k, v in class_to_id.items()}

# --- 1. Robust Loss Function ---
class WeightedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        # Give 20x more weight to Debris (1) than Background (0)
        alpha = torch.where(targets == 1, 0.95, 0.05) 
        return (alpha * (1-pt)**self.gamma * bce).mean()

def save_visualizations(model, loader, device, save_dir, epoch_name):
    model.eval()
    vis_dir = os.path.join(save_dir, 'visualizations', epoch_name)
    os.makedirs(vis_dir, exist_ok=True)
    print(f"Generating visualizations in {vis_dir}...")
    
    with torch.no_grad():
        for i, (image, mask, boxes) in enumerate(tqdm(loader, desc="Visualizing")):
            image = image.to(device)
            
            # --- Prepare Inputs ---
            batch_boxes = []
            batch_labels = []
            for meta in boxes:
                if 'boxes' in meta and len(meta['boxes']) > 0:
                    b = torch.tensor(meta['boxes']).float().to(device)
                    if b.dim() == 1: b = b.unsqueeze(0)
                    batch_boxes.append(b[:, :4])
                    
                    if 'labels' in meta and len(meta['labels']) > 0:
                        batch_labels.append(torch.tensor(meta['labels']).long().to(device))
                    else:
                        batch_labels.append(torch.ones(b.shape[0], dtype=torch.long, device=device))
                else:
                    batch_boxes.append(torch.empty(0, 4).to(device))
                    batch_labels.append(torch.empty(0).long().to(device))
            
            # --- Forward ---
            pred_logits = model(image, batch_boxes, batch_labels)
            pred_prob = torch.sigmoid(pred_logits)
            
            # --- Processing for Display ---
            # 1. Image (Left Panel)
            img_np = image[0].permute(1, 2, 0).cpu().numpy()
            img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # 2. Ground Truth (Middle Panel)
            # Find which class ID is present in the mask
            gt_mask_idx = mask[0].argmax(0).cpu().numpy() # [H, W] values 0-11
            gt_color = np.zeros_like(img_np)
            gt_color[gt_mask_idx > 0] = [0, 0, 255] # Red for objects
            
            # Extract GT Class Names
            unique_gt = np.unique(gt_mask_idx)
            gt_names = []
            for cls_id in unique_gt:
                if cls_id == 0: continue
                # Use class map if available, else just ID
                name = id_to_class.get(cls_id, f"ID {cls_id}")
                gt_names.append(name)
            
            gt_text = ", ".join(gt_names) if gt_names else "Background"

            # 3. Prediction (Right Panel)
            pred_mask_idx = pred_prob[0].argmax(0).cpu().numpy() # [H, W] values 0-11
            pred_color = np.zeros_like(img_np)
            
            unique_pred = np.unique(pred_mask_idx)
            pred_names = []
            for cls_id in unique_pred:
                if cls_id == 0: continue
                pred_color[pred_mask_idx == cls_id] = [0, 255, 0] # Green for prediction
                name = id_to_class.get(cls_id, f"ID {cls_id}")
                pred_names.append(name)

            pred_text = ", ".join(pred_names) if pred_names else "Background"

            # --- Add Text Overlays ---
            # On GT Image (Middle)
            cv2.putText(gt_color, f"Actual: {gt_text}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # On Pred Image (Right)
            cv2.putText(pred_color, f"Pred: {pred_text}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Combine: Input | GT | Prediction
            vis = np.hstack([img_np, gt_color, pred_color])
            cv2.imwrite(os.path.join(vis_dir, f"test_{i:04d}.png"), vis)

def apply_box_jitter(boxes, image_size=1024, noise_scale=20):
    """
    Adds random noise to box coordinates to simulate detector uncertainty.
    boxes: Tensor of shape [N, 4] (x1, y1, x2, y2)
    noise_scale: Max pixels to shift (e.g., +/- 20 pixels)
    """
    if boxes.numel() == 0:
        return boxes
        
    device = boxes.device
    N = boxes.shape[0]
    
    # Generate random noise: integers between -noise_scale and +noise_scale
    noise = torch.randint(-noise_scale, noise_scale + 1, (N, 4), device=device)
    
    # Apply noise
    jittered_boxes = boxes + noise
    
    # Clamp coordinates to ensure they stay inside the image [0, 1024]
    # We use .clamp() so a box doesn't drift off the screen
    jittered_boxes[:, 0] = jittered_boxes[:, 0].clamp(0, image_size) # x1
    jittered_boxes[:, 1] = jittered_boxes[:, 1].clamp(0, image_size) # y1
    jittered_boxes[:, 2] = jittered_boxes[:, 2].clamp(0, image_size) # x2
    jittered_boxes[:, 3] = jittered_boxes[:, 3].clamp(0, image_size) # y2
    
    return jittered_boxes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save_path", type=str, default='')
    args = parser.parse_args()
    opt = Config_SAM(config_path=args.config)
    rand_seed(opt.RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup Logging
    if args.save_path == '':
        opt.MODEL_DIR += f"_{opt.MODEL_NAME}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    else:
        opt.MODEL_DIR = args.save_path
    os.makedirs(opt.MODEL_DIR, exist_ok=True)
    logger = Logger(opt.MODEL_NAME, path=opt.MODEL_DIR)
    
    # Datasets (Reduced workers to prevent RAM crash)
    print("Loading Datasets...")
    train_dataset = DebrisDataset(root_path=opt.DATA_PATH, image_list=os.path.join(opt.IMAGE_LIST_PATH, 'train.txt'),
                                  input_size=opt.INPUT_SIZE, use_augment=True)
    test_dataset = DebrisDataset(root_path=opt.DATA_PATH, image_list=os.path.join(opt.IMAGE_LIST_PATH, 'test.txt'),
                                  input_size=opt.INPUT_SIZE, use_augment=False)

    train_loader = DataLoader(train_dataset, batch_size=opt.TRAIN_BATCHSIZE, shuffle=True, num_workers=0, collate_fn=collate_fn_seq)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn_seq)

    # # Model
    print("Initializing Model...")
    net = SonarSAM(model_name=opt.SAM_NAME, checkpoint=opt.SAM_CHECKPOINT, num_classes=opt.OUTPUT_CHN).to(device)

    if opt.OPTIMIZER == 'ADAM':
        optimizer = torch.optim.Adam(net.parameters(), lr=opt.LEARNING_RATE, weight_decay=opt.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=opt.LEARNING_RATE, momentum=opt.MOMENTUM, weight_decay=opt.WEIGHT_DECAY)
        
    criterion = WeightedFocalLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.EPOCH_NUM)

    best_loss = 999.0
    
    # --- TRAINING LOOP ---
    for epoch in range(opt.EPOCH_NUM):
        net.train()
        epoch_loss = 0
        logger.write_and_print(f'------- Epoch {epoch+1}/{opt.EPOCH_NUM} -------')
        
        for step, (image, mask, boxes) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad()

            batch_boxes = []
            batch_labels = []
            
            # --- ROBUST EXTRACTION WITH REAL LABELS ---
            for meta in boxes:
                if 'boxes' in meta and len(meta['boxes']) > 0:
                    b = torch.tensor(meta['boxes']).float().to(device)

                    # --- NEW CODE: APPLY JITTER ---
                    # Only apply jitter during TRAINING, never validation/testing
                    if net.training:
                        # We jitter by up to 20 pixels (approx 2% of image size)
                        b = apply_box_jitter(b, image_size=opt.INPUT_SIZE, noise_scale=20)
                    # -----------------------------

                    if b.dim() == 1: b = b.unsqueeze(0)
                    batch_boxes.append(b[:, :4])
                    
                    # CHECK FOR REAL LABELS
                    if 'labels' in meta and len(meta['labels']) > 0:
                        l = torch.tensor(meta['labels']).long().to(device)
                        batch_labels.append(l)
                        
                        # DEBUG: Print classes for the first batch only
                        if step == 0:
                            class_names = [id_to_class.get(x.item(), str(x.item())) for x in l]
                            # print(f"\n[DEBUG] Loaded Classes: {class_names}")
                    else:
                        # Only fallback if truly empty
                        batch_labels.append(torch.ones(b.shape[0], dtype=torch.long, device=device))
                else:
                    batch_boxes.append(torch.empty(0, 4).to(device))
                    batch_labels.append(torch.empty(0).long().to(device))
            
            # Forward
            pred_logits = net(image, batch_boxes, batch_labels)
            
            # Loss
            bce = criterion(pred_logits, mask)
            dice = dice_loss(label=mask, mask=pred_logits)
            loss = (bce * 20.0) + dice
            
            if torch.isnan(loss):
                continue
                
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        logger.write_and_print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
        scheduler.step()
        
        # Save Best Model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'state_dict': net.state_dict()}, os.path.join(opt.MODEL_DIR, f"{opt.MODEL_NAME}_best.pth"))

    # --- FINAL TEST & VISUALIZATION ---
    print("\n--- Running Final Visualization on Test Set ---")
    best_ckpt = os.path.join(opt.MODEL_DIR, f"{opt.MODEL_NAME}_best.pth")
    checkpoint = torch.load(best_ckpt)
    net.load_state_dict(checkpoint['state_dict'])
    
    save_visualizations(net, test_loader, device, opt.MODEL_DIR, "final_test_results")
    print(f"\nAll Done! Visualizations saved to: {os.path.join(opt.MODEL_DIR, 'visualizations', 'final_test_results')}")

if __name__ == "__main__":
    main()