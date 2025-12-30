#!/usr/bin/env python
# Simple script to enhance a single image or folder of images using pretrained Retinexformer

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import argparse
from tqdm import tqdm
import cv2
from glob import glob
from natsort import natsorted

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models import create_model
from basicsr.utils.options import parse

def load_model(opt_path, weights_path, device='cpu'):
    """Load the pretrained model"""
    # Parse options
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    
    # Create model
    model = create_model(opt).net_g
    
    # Load weights
    checkpoint = torch.load(weights_path, map_location=device)
    try:
        model.load_state_dict(checkpoint['params'])
    except:
        # Handle DataParallel wrapped weights
        new_checkpoint = {}
        for k in checkpoint['params']:
            new_checkpoint['module.' + k] = checkpoint['params'][k]
        model.load_state_dict(new_checkpoint)
    
    model.to(device)
    model.eval()
    print(f"Model loaded from {weights_path}")
    
    return model

def enhance_image(model, image_path, output_path, device='cpu', factor=4):
    """Enhance a single image"""
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Padding for multiples of factor
    b, c, h, w = img_tensor.shape
    H = ((h + factor) // factor) * factor
    W = ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    img_tensor = F.pad(img_tensor, (0, padw, 0, padh), 'reflect')
    
    # Inference
    with torch.inference_mode():
        restored = model(img_tensor)
    
    # Unpad and convert back
    restored = restored[:, :, :h, :w]
    restored = torch.clamp(restored, 0, 1)
    restored = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Convert back to BGR and save
    restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
    restored = (restored * 255.0).astype(np.uint8)
    cv2.imwrite(output_path, restored)
    
    return restored

def main():
    parser = argparse.ArgumentParser(description='Enhance images using pretrained Retinexformer')
    parser.add_argument('--input', type=str, required=True, 
                        help='Input image path or folder containing images')
    parser.add_argument('--output', type=str, required=True,
                        help='Output folder for enhanced images')
    parser.add_argument('--target', type=str, default=None,
                        help='Target/ground truth folder for PSNR/SSIM calculation (optional)')
    parser.add_argument('--weights', type=str, 
                        default='../pretrained_weights/LOL_v1.pth',
                        help='Path to pretrained weights')
    parser.add_argument('--opt', type=str,
                        default='../Options/RetinexFormer_LOL_v1.yml',
                        help='Path to options YAML file')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available (default: CPU)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using GPU')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = torch.device('cpu')
        print('Using CPU')
    
    # Load model
    model = load_model(args.opt, args.weights, device)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get input files
    if os.path.isfile(args.input):
        image_paths = [args.input]
    elif os.path.isdir(args.input):
        image_paths = natsorted(
            glob(os.path.join(args.input, '*.png')) + 
            glob(os.path.join(args.input, '*.jpg')) +
            glob(os.path.join(args.input, '*.jpeg')) +
            glob(os.path.join(args.input, '*.bmp'))
        )
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return
    
    if len(image_paths) == 0:
        print(f"No images found in {args.input}")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    # Get target images if specified
    target_paths = None
    if args.target is not None:
        if os.path.isfile(args.target):
            target_paths = [args.target]
        elif os.path.isdir(args.target):
            target_paths = natsorted(
                glob(os.path.join(args.target, '*.png')) + 
                glob(os.path.join(args.target, '*.jpg')) +
                glob(os.path.join(args.target, '*.jpeg')) +
                glob(os.path.join(args.target, '*.bmp'))
            )
        
        if len(target_paths) != len(image_paths):
            print(f"Warning: Number of target images ({len(target_paths)}) doesn't match input images ({len(image_paths)})")
            print("PSNR/SSIM calculation will be skipped.")
            target_paths = None
        else:
            print(f"Target images found. Will calculate PSNR and SSIM.")
    
    # Lists to store metrics
    psnr_list = []
    ssim_list = []
    
    # Process images
    for idx, img_path in enumerate(tqdm(image_paths)):
        filename = os.path.basename(img_path)
        output_path = os.path.join(args.output, filename)
        
        try:
            restored = enhance_image(model, img_path, output_path, device)
            
            # Calculate metrics if target is provided
            if target_paths is not None:
                target_path = target_paths[idx]
                target = cv2.imread(target_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
                target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
                
                # Ensure same size
                if target.shape != restored.shape:
                    print(f"Warning: Size mismatch for {filename}. Skipping metrics.")
                else:
                    # Convert to uint8 for metrics calculation
                    restored_uint8 = (restored * 255.0).astype(np.uint8)
                    target_uint8 = (target * 255.0).astype(np.uint8)
                    
                    # Import metrics functions
                    import sys
                    utils_path = os.path.join(project_root, 'Enhancement')
                    if utils_path not in sys.path:
                        sys.path.insert(0, utils_path)
                    import utils
                    
                    psnr_val = utils.calculate_psnr(target_uint8, restored_uint8)
                    ssim_val = utils.calculate_ssim(target_uint8, restored_uint8)
                    psnr_list.append(psnr_val)
                    ssim_list.append(ssim_val)
                    
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    print(f"Done! Enhanced images saved to {args.output}")
    
    # Print average metrics if calculated
    if len(psnr_list) > 0:
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        print(f"\n{'='*50}")
        print(f"Average PSNR: {avg_psnr:.4f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"{'='*50}")

if __name__ == '__main__':
    main()
