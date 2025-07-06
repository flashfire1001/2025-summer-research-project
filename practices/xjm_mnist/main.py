
# ============ main.py ============
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sampler import MNISTSampler
from model import MNISTUNet
from trainer import CFGTrainer
from ode_path import LinearAlpha, LinearBeta, GaussianConditionalProbabilityPath, CFGVectorFieldODE
from simulator import EulerSimulator
from common_config import device


# --- Parameters ---
num_rows = 3
num_cols = 3
num_timesteps = 5
samples_per_class = 3
guidance_scales = [1.0, 3.0, 5.0]
batch_size = 250
num_epochs = 1  # For quick test, reduce to ~10–20



# --- Sampler and Path ---
sampler = MNISTSampler().to(device)
path = GaussianConditionalProbabilityPath(
    p_data=sampler,
    p_simple_shape=[1, 32, 32], # size issue???
    alpha=LinearAlpha(),
    beta=LinearBeta()
).to(device)

# --- Model ---
unet = MNISTUNet(
    channels=[32, 64, 128],
    nres=2,
    tdim=40,
    ydim=40
).to(device)

# --- Training ---
trainer = CFGTrainer(path=path, model=unet, eta=0.1)
loss_record = trainer.train(num_epochs=num_epochs, device=device, lr=1e-3, batch_size=batch_size)

# --- Visualization ---
fig, axes = plt.subplots(len(guidance_scales), 1, figsize=(10, 10 * len(guidance_scales)))
for idx, w in enumerate(guidance_scales):
    ode = CFGVectorFieldODE(unet, guidance_scale=w)
    simulator = EulerSimulator(ode)

    y = torch.tensor(list(range(10)) + [10], dtype=torch.int64).repeat_interleave(samples_per_class).to(device)
    x0, _ = path.p_simple.sample(y.shape[0])
    ts = torch.linspace(0, 1, num_timesteps).view(1, -1, 1, 1, 1).expand(y.shape[0], -1, 1, 1, 1).to(device)
    x1 = simulator.simulate(x0, ts, y=y)
    images = x1
    images = images.view(11, samples_per_class, 1, 32, 32)  # [classes, samples, C, H, W]
    images = images.transpose(0, 1)  # [11, 3, C, H, W]
    images = images.reshape(-1, 1, 32, 32)  # [33, C, H, W]

    grid = make_grid(images, nrow=11, normalize=True, value_range=(-1, 1))
    axes[idx].imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray") # (H, W, C) for imshow requirement
    axes[idx].axis("off")
    axes[idx].set_title(f"Guidance: $w={w:.1f}$", fontsize=20)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(range(num_epochs), loss_record, label='Training Loss')
plt.xlabel('Epoch',fontsize = 14)
plt.ylabel('Loss', fontsize = 14)
plt.title('Loss Function over Epochs',fontsize = 16)
plt.legend(fontsize = 12)
plt.show()

# torch.to_numpy()

# how to graph a loss function curve
# how to to use make grid methods