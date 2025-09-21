import numpy as np
import torch
from pytorch_msssim import ssim

def retrieve_imgs(model, input_img, train_loader, DEVICE='cuda' if torch.cuda.is_available() else 'cpu', num_imgs=3):
    model.eval()
    input_img = input_img.to(DEVICE)
    
    with torch.no_grad():
        input_bottleneck = model.encoder(input_img.unsqueeze(0))
        decoded_input = model.decoder(input_bottleneck)
    
    all_scores = []
    all_imgs = []

    with torch.no_grad():
        for imgs, _ in train_loader:
            imgs = imgs.to(DEVICE)
            batch_bottleneck = model.encoder(imgs)

            for i in range(batch_bottleneck.size(0)):
                score = ssim(input_bottleneck, batch_bottleneck[i:i+1], data_range=1.0)
                all_scores.append(score.item())
                all_imgs.append(imgs[i])

    sorted_indices = np.argsort(all_scores)[::-1][:num_imgs]
    top_imgs = [all_imgs[i] for i in sorted_indices]

    return  input_img.cpu(),input_bottleneck.squeeze(0).cpu().numpy(),[img.cpu() for img in top_imgs], decoded_input.squeeze(0).cpu().numpy()  

