import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset

def prepare_images(test_folder, inject_folder, transform, base_loader, batch_size=64):
    test_img = Image.open(test_folder).convert("RGB")
    test_tensor = transform(test_img)

    inject_tensors = []
    for fname in os.listdir(inject_folder):
        fpath = os.path.join(inject_folder, fname)
        if os.path.isfile(fpath):
            try:
                img = Image.open(fpath).convert("RGB")
                inject_tensors.append(transform(img))
            except Exception as e:
                print(f"Skipping {fname}: {e}")

    if not inject_tensors:
        print("⚠️ No valid inject images found!")
    
    inject_dataset = TensorDataset(torch.stack(inject_tensors)) if inject_tensors else None

    base_dataset = base_loader.dataset
    if inject_dataset:
        combined_dataset = ConcatDataset([base_dataset, inject_dataset])
    else:
        combined_dataset = base_dataset

    new_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    return test_tensor, new_loader