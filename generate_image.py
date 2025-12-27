"""
SpinNet Image Generation
------------------------
Generate MNIST-style images by sampling from a trained image model.

Usage:
    python generate_image.py --ckpt experiments/out-mnist-hadamard/ckpt.pt
    python generate_image.py --ckpt ... --num_images 10 --output_dir generated/
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from src.model import SpinNetConfig, SpinNet


def tokens_to_image(tokens, patch_size=4, grid_size=7):
    """Convert 49 patch tokens back to a 28x28 image."""
    img = np.zeros((grid_size * patch_size, grid_size * patch_size), dtype=np.uint8)
    
    for idx, token in enumerate(tokens[:grid_size * grid_size]):
        i, j = idx // grid_size, idx % grid_size
        # Fill patch with the token value (grayscale)
        img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = token
    
    return img


def generate_images(model, device, num_images=9, temperature=1.0, top_k=None):
    """Generate images autoregressively from the separator token."""
    model.eval()
    images = []
    
    SEP_TOKEN = 256  # Separator token signals start of new image
    TOKENS_PER_IMAGE = 49
    
    with torch.no_grad():
        for i in range(num_images):
            # Start with separator token
            x = torch.tensor([[SEP_TOKEN]], device=device)
            
            # Generate 49 patch tokens
            for _ in range(TOKENS_PER_IMAGE):
                # Get logits for next token
                logits, _ = model(x)
                logits = logits[:, -1, :] / temperature
                
                # Only consider grayscale tokens (0-255), not separator
                logits[:, 256] = float('-inf')
                
                # Top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, next_token], dim=1)
            
            # Convert tokens to image (skip the initial separator)
            tokens = x[0, 1:].cpu().numpy()
            img = tokens_to_image(tokens)
            images.append(img)
            
            if (i + 1) % 3 == 0:
                print(f'Generated {i+1}/{num_images} images')
    
    return images


def create_grid(images, grid_size=3):
    """Create a grid of images."""
    n = len(images)
    rows = (n + grid_size - 1) // grid_size
    
    img_size = images[0].shape[0]
    grid = np.ones((rows * img_size + (rows-1)*2, 
                    grid_size * img_size + (grid_size-1)*2), dtype=np.uint8) * 128
    
    for idx, img in enumerate(images):
        i, j = idx // grid_size, idx % grid_size
        y, x = i * (img_size + 2), j * (img_size + 2)
        grid[y:y+img_size, x:x+img_size] = img
    
    return grid


def main():
    parser = argparse.ArgumentParser(description='Generate images from SpinNet')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--num_images', type=int, default=9, help='Number of images')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temp')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--output_dir', type=str, default='generated', help='Output dir')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f'Loading model from {args.ckpt}...')
    print(f'Device: {device}')
    
    # Load model
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = SpinNetConfig(**checkpoint['model_args'])
    
    print(f'Model: {config.n_layer}L × {config.n_head}H × {config.n_embd}D')
    print(f'Algebra: {getattr(config, "algebra", "octonion")} | Hash: {getattr(config, "hash_embeddings", False)}')
    
    model = SpinNet(config)
    
    # Load weights
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Generate
    print(f'\nGenerating {args.num_images} images...')
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device == 'cuda' else torch.no_grad():
        images = generate_images(model, device, args.num_images, args.temperature, args.top_k)
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save individual images
    for i, img in enumerate(images):
        pil_img = Image.fromarray(img, mode='L')
        pil_img.save(os.path.join(args.output_dir, f'image_{i:03d}.png'))
    
    # Save grid
    grid = create_grid(images)
    grid_img = Image.fromarray(grid, mode='L')
    grid_path = os.path.join(args.output_dir, 'grid.png')
    grid_img.save(grid_path)
    
    print(f'\nSaved {len(images)} images to {args.output_dir}/')
    print(f'Grid: {grid_path}')


if __name__ == '__main__':
    main()
