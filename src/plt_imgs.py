import numpy as np
import torch
import matplotlib.pyplot as plt

def plt_imgs(input_img=None, bottleneck_tensor=None, retrieved_imgs=None, decoded_bn=None):

    n_retrieved = len(retrieved_imgs) if retrieved_imgs is not None else 0
    total_cols = 1 + (1 if bottleneck_tensor is not None else 0) + n_retrieved + (1 if decoded_bn is not None else 0)

    plt.figure(figsize=(3*total_cols, 4))

    def prepare_image(img):
        if isinstance(img, torch.Tensor):
            img = img.cpu().detach().numpy()
        img = np.squeeze(img)

        if img.ndim == 3 and img.shape[0] in [3,4]:
            img = np.transpose(img, (1,2,0))

        img = np.clip(img, 0, 1)
        return img

    def plot_img(img, title, pos):
        plt.subplot(1, total_cols, pos)
        plt.imshow(prepare_image(img))
        plt.title(title)
        plt.axis('off')

    col_idx = 1
    if input_img is not None:
        plot_img(input_img, "Input Image", col_idx)
        col_idx += 1

    if bottleneck_tensor is not None:
        if isinstance(bottleneck_tensor, torch.Tensor):
            bottleneck_tensor = bottleneck_tensor.cpu().detach().numpy()
        C, H, W = bottleneck_tensor.shape
        side = int(np.ceil(np.sqrt(C)))
        grid = np.zeros((side*H, side*W))
        for idx in range(C):
            row = idx // side
            col = idx % side
            grid[row*H:(row+1)*H, col*W:(col+1)*W] = bottleneck_tensor[idx]
        plt.subplot(1, total_cols, col_idx)
        plt.imshow(grid, cmap='viridis')
        plt.title("Bottleneck")
        plt.axis('off')
        col_idx += 1

    if retrieved_imgs is not None:
        for i, img in enumerate(retrieved_imgs):
            plot_img(img, f"Retrieved {i+1}", col_idx)
            col_idx += 1

    if decoded_bn is not None:
        plot_img(decoded_bn, "Decoded", col_idx)

    plt.tight_layout()
    plt.show()
