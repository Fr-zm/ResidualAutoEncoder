import torch
import numpy as np
from pytorch_msssim import SSIM

def return_ssim(model, loader, DEVICE='cuda' if torch.cuda.is_available() else 'cpu'):
    ssim_module = SSIM(data_range=1.0, size_average=True, channel=3).to(DEVICE)
    model.eval()
    all_ssim = []

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)

            s = ssim_module(outputs, imgs)
            all_ssim.append(s.item())

    mean_ssim = np.mean(all_ssim)
    return mean_ssim