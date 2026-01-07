import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging

# Imports for SAM 2
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from sam2.build_sam import _load_checkpoint
import math

def build_sam2_local(config_path, checkpoint_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at: {config_path}")

    config_dir = os.path.dirname(os.path.abspath(config_path))
    config_name = os.path.basename(config_path)
    
    if config_name.endswith('.yaml'):
        config_name = config_name[:-5]
        
    GlobalHydra.instance().clear()
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
        
    try:
        if 'model' in cfg and cfg.model is not None:
            model = instantiate(cfg.model)
        else:
            model = instantiate(cfg)
    except Exception as e:
        print(f"Error instantiating model from config: {e}")
        model = instantiate(cfg)
    
    _load_checkpoint(model, checkpoint_path)
    return model

class SonarSAM(nn.Module):
    def __init__(self, model_name, checkpoint, num_classes=12, 
                 is_finetune_image_encoder=False,
                 use_adaptation=False, adaptation_type='LORA', 
                 head_type='custom', reduction=4, upsample_times=2, groups=4, rank=4):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes

        # Path Setup
        ckpt_dir = os.path.dirname(os.path.abspath(checkpoint))
        if 'tiny' in model_name or 'mobile' in model_name:
            cfg_name = "sam2.1_hiera_t.yaml" 
        elif 'small' in model_name or 'base' in model_name:
            cfg_name = "sam2.1_hiera_s.yaml"
        else:
            cfg_name = "sam2.1_hiera_l.yaml"

        local_config_path = os.path.join(ckpt_dir, cfg_name)
        print(f"Loading SAM 2 Config from: {local_config_path}")

        try:
            self.sam = build_sam2_local(local_config_path, checkpoint)
        except Exception as e:
            print(f"CRITICAL ERROR building SAM 2: {e}")
            raise e
        
        # # Freeze Image Encoder
        if not is_finetune_image_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False

    def forward(self, images, boxes=None, labels=None):
        batch_size = images.shape[0]
        
        # 1. Image Encoder
        backbone_out = self.sam.forward_image(images)
        
        if isinstance(backbone_out, dict):
            if 'image_embed' not in backbone_out and 'vision_features' in backbone_out:
                backbone_out['image_embed'] = backbone_out['vision_features']
            if 'high_res_feats' not in backbone_out and 'backbone_fpn' in backbone_out:
                backbone_out['high_res_feats'] = backbone_out['backbone_fpn'][:2]

        batch_outputs = []
        
        for i in range(batch_size):
            # Check for boxes
            current_boxes = None
            current_labels = None
            
            if boxes is not None and len(boxes) > i and torch.is_tensor(boxes[i]) and boxes[i].numel() > 0:
                current_boxes = boxes[i].unsqueeze(1) # [N, 1, 4]
                if labels is not None:
                    current_labels = labels[i]
            else:
                # No boxes? Return "All Background" tensor safely
                # Background (+2.0), Others (-2.0)
                semantic_output = torch.full((1, self.num_classes, images.shape[2], images.shape[3]), 
                                          -2.0, device=images.device)
                semantic_output[:, 0, :, :] = 2.0
                batch_outputs.append(semantic_output)
                continue

            # Prepare Inputs
            num_prompts = current_boxes.shape[0]
            img_embed = backbone_out['image_embed'][i].unsqueeze(0).expand(num_prompts, -1, -1, -1)
            hi_res = [f[i].unsqueeze(0).expand(num_prompts, -1, -1, -1) for f in backbone_out['high_res_feats']]

            # Prompt Encoder
            sparse_embeddings, dense_embeddings = self.sam.sam_prompt_encoder(
                points=None, boxes=current_boxes, masks=None,
            )

            # Mask Decoder
            low_res_masks, iou_predictions, *rest = self.sam.sam_mask_decoder(
                image_embeddings=img_embed,
                image_pe=self.sam.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True, 
                repeat_image=False,
                high_res_features=hi_res
            )
            
            # Post-Process: Select best mask and upscale
            if low_res_masks.dim() == 3:
                 low_res_masks = low_res_masks.unsqueeze(1)
            elif low_res_masks.dim() == 4 and low_res_masks.shape[1] == 3:
                low_res_masks = low_res_masks[:, 0:1, :, :]
                
            # Upscale shape: [N_Boxes, 1, H, W]
            upscaled_masks = F.interpolate(
                low_res_masks, size=(images.shape[2], images.shape[3]),
                mode="bilinear", align_corners=False,
            )

            # --- FIX: SAFE ACCUMULATION (No In-Place Ops) ---
            # 1. Initialize lists to hold masks for each class
            # Default values: Background=+2.0, Debris=-2.0
            
            # Dictionary: Class ID -> List of tensors [1, H, W]
            class_masks_list = {c: [torch.full((1, images.shape[2], images.shape[3]), -2.0, device=images.device)] 
                                for c in range(1, self.num_classes)}
            
            # List for Background (starts with +2.0)
            bg_masks_list = [torch.full((1, images.shape[2], images.shape[3]), 2.0, device=images.device)]

            if current_labels is not None:
                for b_idx, label in enumerate(current_labels):
                    label_idx = int(label.item())
                    
                    if 0 < label_idx < self.num_classes:
                        mask_logit = upscaled_masks[b_idx] # Shape [1, H, W]
                        
                        # Add to the specific class list
                        class_masks_list[label_idx].append(mask_logit)
                        
                        # Add negative logit to background list (to suppress BG)
                        bg_masks_list.append(-mask_logit)

            # 2. Reduce (Max for Objects, Min for Background)
            final_channels = []
            
            # Channel 0: Background -> Take MIN of all contributions
            bg_stack = torch.cat(bg_masks_list, dim=0) # Stack along dim 0
            bg_final = torch.amin(bg_stack, dim=0, keepdim=True) # [1, H, W]
            final_channels.append(bg_final)

            # Channels 1-11: Objects -> Take MAX of all contributions
            for c in range(1, self.num_classes):
                c_stack = torch.cat(class_masks_list[c], dim=0)
                c_final = torch.amax(c_stack, dim=0, keepdim=True) # [1, H, W]
                final_channels.append(c_final)

            # 3. Stack channels -> [1, 12, H, W]
            semantic_output = torch.cat(final_channels, dim=0).unsqueeze(0)
            
            batch_outputs.append(semantic_output)

        final_output = torch.cat(batch_outputs, dim=0)
        return final_output

class ModelWithLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.bcewithlogit = nn.BCEWithLogitsLoss(reduction='mean')
        from model.loss_functions import dice_loss
        self.dice_loss = dice_loss

    def forward(self, images, masks, boxes=None):
        pred_masks = self.model(images, boxes)
        bce_loss = self.bcewithlogit(input=pred_masks, target=masks)
        dice = self.dice_loss(label=masks, mask=pred_masks)
        loss = bce_loss + dice
        return loss, pred_masks