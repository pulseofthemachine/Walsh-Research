"""
MNIST Data Preparation for SpinNet Image Generation
----------------------------------------------------
Converts MNIST images into tokenized sequences for autoregressive generation.

Each 28x28 image is:
1. Split into 4x4 patches (7x7 = 49 patches per image)
2. Each patch is averaged to a single token (0-255)
3. Saved as binary token sequences

Usage:
    python data/mnist/prepare.py
"""

import os
import numpy as np
from torchvision import datasets

def image_to_patch_tokens(image, patch_size=4):
    """Convert a 28x28 image to 49 patch tokens."""
    image = np.array(image)
    H, W = image.shape
    nH, nW = H // patch_size, W // patch_size
    
    tokens = []
    for i in range(nH):
        for j in range(nW):
            patch = image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            tokens.append(int(patch.mean()))
    
    return np.array(tokens, dtype=np.uint16)

def prepare_dataset(data_dir, output_dir):
    """Prepare MNIST as tokenized sequences."""
    
    print('Downloading MNIST via torchvision...')
    train_dataset = datasets.MNIST(data_dir, train=True, download=True)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True)
    
    print(f'Train: {len(train_dataset)} images')
    print(f'Test: {len(test_dataset)} images')
    
    print('Tokenizing images...')
    SEP_TOKEN = 256
    
    # Train
    train_data = []
    for i, (img, _) in enumerate(train_dataset):
        train_data.extend(image_to_patch_tokens(img).tolist())
        train_data.append(SEP_TOKEN)
        if i % 10000 == 0:
            print(f'  Train: {i}/{len(train_dataset)}')
    
    # Val
    val_data = []
    for img, _ in test_dataset:
        val_data.extend(image_to_patch_tokens(img).tolist())
        val_data.append(SEP_TOKEN)
    
    train_data = np.array(train_data, dtype=np.uint16)
    val_data = np.array(val_data, dtype=np.uint16)
    
    print(f'Train tokens: {len(train_data):,}')
    print(f'Val tokens: {len(val_data):,}')
    print(f'Vocab size: 257 (0-255 grayscale + 1 separator)')
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    train_data.tofile(os.path.join(output_dir, 'train.bin'))
    val_data.tofile(os.path.join(output_dir, 'val.bin'))
    
    # Metadata
    import pickle
    meta = {
        'vocab_size': 257,
        'tokens_per_image': 50,  # 49 patches + 1 separator
        'patch_size': 4,
        'image_size': 28,
        'train_images': len(train_dataset),
        'val_images': len(test_dataset),
    }
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print(f'\nSaved to {output_dir}/')

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prepare_dataset(script_dir, script_dir)
    print('\nDone! Run training with:')
    print('  python train.py config/train_mnist_image.py')
