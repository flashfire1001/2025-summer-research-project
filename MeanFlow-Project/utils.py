import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from typing import List
import torch.nn.functional as F


def add_gaussian_noise(image: torch.Tensor, eta: float = 0.5) -> torch.Tensor:
    # image: (c, h, w) or (bs, c, h, w)

    single = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        single = True

    eta_val = torch.empty(1).uniform_(eta, 0.5).item()
    noise = torch.randn_like(image)
    noised = image + eta_val * noise
    # clipped = torch.clamp(noised, min=0.0, max=1.0)

    return noised.squeeze(0) if single else noised


def add_sp_noise(image: torch.Tensor, amount: float = 0.2) -> torch.Tensor:
    # image: (c, h, w) or (bs, c, h, w)
    single = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        single = True
    mask = torch.rand_like(image)
    image[mask < amount / 2] = 0.0
    image[(mask >= amount / 2) & (mask < amount)] = 1.0
    return image.squeeze(0) if single else image


def add_block_mask(image: torch.Tensor, size: float = 10.0) -> torch.Tensor:
    # image: (c, h, w) or (bs, c, h, w)
    single = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        single = True

    B, C, H, W = image.shape
    size = int(size)

    # the left-most part to be blocked, of shape(bs, 1)
    x_start = torch.randint(0, H - size, (B, ), device=image.device).view(-1, 1)
    y_start = torch.randint(0, W - size, (B, ), device=image.device).view(-1, 1)

    # the range to be blocked, of shape(1, size)
    x_range = torch.arange(size, device=image.device).view(1, -1)
    y_range = torch.arange(size, device=image.device).view(1, -1)

    # the bolck part, of shape(bs, size)
    x = x_start + x_range
    y = y_start + y_range

    # construct the indices
    X = x[:, :, None].expand(-1, -1, size)  # (bs, size, 1) -> (bs, size, size)
    Y = y[:, None, :].expand(-1, size, -1)  # (bs, 1, size) -> (bs, size, size)
    batch_idx = torch.arange(B, device=image.device).view(-1, 1, 1).expand(-1, size, size)

    image[batch_idx, :, X, Y] = 0.5

    return image.squeeze(0) if single else image


def basic_visualize(tensor, savepath, title=None, max_images=10, img_size=3):
    """
    visualize images of shape of (C, H, W) or (B, C, H, W)
    if of multiple images, show at most first 8 images
    """

    # check whether tensor
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    else:
        raise TypeError("The input should be torch.Tensor")

    if tensor.dim() == 3:
        imgs = tensor.unsqueeze(0)
    elif tensor.ndim == 4:
        imgs = tensor
    else:
        raise ValueError(f"Unsupported dim: {tensor.shape}")

    B, C, H, W = imgs.shape
    num_show = min(B, max_images)

    # convert from [-1, 1] to [0, 1]
    # imgs = (imgs + 1) / 2
    imgs = imgs.clamp(0, 1)

    # create a row of image of num_show cols, with each image of size 3 * 3
    fig, axes = plt.subplots(1, num_show, figsize=(img_size * num_show, img_size))

    if num_show == 1:
        axes = [axes]

    for i in range(num_show):
        img = imgs[i]
        if C == 1:
            img = img[0]
            axes[i].imshow(img, cmap='gray')
        else:
            img = img.permute(1, 2, 0)  # C, H, W -> H, W, C
            axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"{title or ''} [{i}]")

    plt.tight_layout()
    plt.savefig(savepath, dpi=300)


def ewma_smooth(data_array, alpha=0.2):
    """
    Applies Exponentially Weighted Moving Average (EWMA) smoothing to the data.
    alpha is the smoothing factor (0 < alpha <= 1).
    Smaller alpha values result in more smoothing.
    """
    smoothed_data = np.zeros_like(data_array, dtype=float)
    if len(data_array) > 0:
        smoothed_data[0] = data_array[0] # Initialize with the first value
        for i in range(1, len(data_array)):
            smoothed_data[i] = alpha * data_array[i] + (1 - alpha) * smoothed_data[i-1]
    return smoothed_data


root_dir = Path("results")

def draw_loss_curve(loss_history: List[float], n_steps: int, batch_size: int, project_name: str):
    save_path = root_dir / project_name / f"{project_name}_{n_steps}_steps_{batch_size}_bs_loss_curve.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    losses = np.array(loss_history)
    smoothed_losses = ewma_smooth(losses)
    
    fig, ax = plt.subplots() 
    ax.plot(losses, color='lightgray', label='Raw Loss')
    ax.plot(smoothed_losses, color='blue', label=f'Smoothed Loss (EWMA)')
    ax.set_title(f"{project_name.upper()} Loss Curve with {n_steps} steps")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def compute_psnr(clean_img: torch.Tensor, restore_img: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    # clean_img, restore_img: (bs, 1, 32, 32)
    mse = F.mse_loss(restore_img, clean_img, reduction='none')  # shape: (bs, 1, 32, 32)
    mse = mse.view(mse.size(0), -1).mean(dim=1)  # shape: (bs,)
    psnr = 10 * torch.log10(max_val ** 2 / (mse + 1e-10))  # avoid divide-by-zero
    return torch.mean(psnr)  # shape: (bs,)


def should_test(step: int):
    if step % 400 == 0:
            return True
    return False
