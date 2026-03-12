# import pandas as pd
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# from concurrent.futures import ProcessPoolExecutor

# def process_image(path):
#     try:
#         # Open and normalize
#         img = np.array(Image.open(path).convert('RGB')) / 255.0
#         # Return mean and mean of squares for the 3 channels
#         return np.mean(img, axis=(0, 1)), np.mean(img**2, axis=(0, 1))
#     except Exception:
#         return None, None

# if __name__ == "__main__":
#     df = pd.read_csv('C:/FYP/stages/initial_layer_dataset.csv')
#     paths = df['img_path'].tolist()
    
#     sums = np.zeros(3)
#     sq_sums = np.zeros(3)
#     count = 0

#     # Use max_workers=None to use all available CPU cores
#     with ProcessPoolExecutor(max_workers=10) as executor:
#         results = list(tqdm(executor.map(process_image, paths), total=len(paths)))

#     for m, s2 in results:
#         if m is not None:
#             sums += m
#             sq_sums += s2
#             count += 1

#     final_mean = sums / count
#     final_std = np.sqrt((sq_sums / count) - (final_mean**2))

#     print(f"MEAN: {final_mean.tolist()}")
#     print(f"STD: {final_std.tolist()}")

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from clean_src.data.dataset import ParametersDataset

def calculate_global_stats(csv_path, root_dir):
    # 1. Initialize the dataset exactly as the paper intends
    # We use ToTensor() to get the [0, 1] range.
    dataset = ParametersDataset(
        csv_file=csv_path,
        root_dir=root_dir,
        image_dim=(320, 320),
        post_crop_transform=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ]),
        per_img_normalisation=False # Crucial: keep False for global stats
    )

    # 2. Use high num_workers for speed
    loader = DataLoader(
        dataset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=8, # Adjust based on your CPU cores
        pin_memory=True
    )

    mean_sum = torch.zeros(3)
    std_sum = torch.zeros(3)
    total_samples = 0

    print("Iterating through cropped dataset...")
    for images, _ in tqdm(loader):
        batch_samples = images.size(0)
        # Flatten H and W: [B, C, H, W] -> [B, C, H*W]
        # Calculate mean and std per image in the batch
        # dim=(2,3) reduces the H and W dimensions
        batch_means = images.mean(dim=(2, 3)) # [batch, 3]
        batch_stds = images.std(dim=(2, 3))   # [batch, 3]
        
        # Sum them up to average later
        mean_sum += batch_means.sum(dim=0)
        std_sum += batch_stds.sum(dim=0)
        
        total_samples += batch_samples

    final_mean = mean_sum / total_samples
    final_std = std_sum / total_samples

    return final_mean, final_std

if __name__ == "__main__":
    CSV_PATH = 'C:/FYP/stages/initial_layer_dataset.csv' # stage 1
    # CSV_PATH = 'C:/FYP/phase1/full_dataset.csv'
    ROOT_DIR = 'C:/FYP/' # Or wherever the base path is
    
    m, s = calculate_global_stats(CSV_PATH, ROOT_DIR)
    
    print(f"\nDATASET_MEAN = {m.tolist()}")
    print(f"DATASET_STD = {s.tolist()}")