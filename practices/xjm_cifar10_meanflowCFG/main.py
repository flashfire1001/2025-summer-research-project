
# ============ main.py ============
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sampler import CIFARSampler,TimeSampler
from model import CIFARUNet
from trainer import CFGTrainer
from ode_path import LinearAlpha, LinearBeta, GaussianConditionalProbabilityPath
from simulator import MeanFlowSimulator
from common_config import device, load_model

# --- Parameters ---

num_timesteps = 2
samples_per_class = 3
guidance_scales = [2.0, 3.0, 4.0, 5.0]
batch_size = 128
num_epochs = 2 # For quick test, reduce to ~2-10
num_classes = 11


# --- Sampler and Path ---
sampler = CIFARSampler().to(device)
TimeSampler = TimeSampler().to(device)
path = GaussianConditionalProbabilityPath(
    p_data=sampler,
    p_simple_shape=[3, 32, 32],
    alpha=LinearAlpha(),
    beta=LinearBeta()
).to(device)

# --- Model ---
unet = CIFARUNet(
    channels=[128, 256, 256,256],
    nres=2,
    tdim=64,
    ydim=64
).to(device)

if __name__ == '__main__':

    generated_images_list = []
    # setup DataLoader with num_workers=4 and run training loop here
    # depict the cifar10 to generate
    fig, axes = plt.subplots(len(guidance_scales), 1, figsize=(10, 5 * len(guidance_scales)))

    # --- Training ---
    for idx , w in enumerate(guidance_scales):
        print(f"training the model with guidance scale {w}")
        trainer = CFGTrainer(path=path,
                             model=unet,
                             eta=0.1, guidance_scale=w,
                             t_sampler=TimeSampler)
        loss_record = trainer.train(num_epochs=num_epochs,
                                    device=device,
                                    lr=1e-4,
                                    batch_size=batch_size)

        # --- Loss Curve ---
        plt.figure(figsize=(10, 7))
        plt.plot(range(100, num_epochs), loss_record[100:], label="loss_function")
        plt.title(f"loss track during training, guidance scale{w}", fontsize=16)
        plt.xlabel("epoch", fontsize=14)
        plt.ylabel("loss", fontsize=14)
        plt.legend(fontsize=12)
        plt.show()
        plt.close()

        # --- Generate Images ---
        # Apply EMA before sampling(simulation)
        trainer.ema.apply_shadow()
        simulator = MeanFlowSimulator(trainer.model)  #now it's ema-applied

        y = torch.tensor(list(range(10)) + [10], dtype=torch.int64).repeat_interleave(samples_per_class).to(device)
        x0, _ = path.p_simple.sample(y.shape[0])
        ts = torch.linspace(0, 1, num_timesteps).view(1, -1, 1, 1, 1).expand(y.shape[0], -1, 1, 1, 1).to(device)
        x1 = simulator.simulate(x0, ts, y=y)

        images = x1.reshape(num_classes, samples_per_class, 3, 32, 32)
        images = images.permute((1, 0, 2, 3, 4))  # [samples_per_class, num_classes, C, H, W]
        images = images.reshape(-1, 3, 32, 32)

        generated_images_list.append((w,images))

        trainer.ema.restore()

    # --- Final Combined Visualization ---
    fig, axes = plt.subplots(len(guidance_scales), 1, figsize=(10, 5 * len(guidance_scales)))
    if len(guidance_scales) == 1:
        axes = [axes]  # make iterable if only 1 axis

    for ax, (w, images) in zip(axes, generated_images_list):
        grid = make_grid(images, nrow=num_classes, normalize=True, value_range=(-1, 1))
        ax.imshow(grid.permute(1, 2, 0).cpu())
        ax.axis("off")
        ax.set_title(f"Guidance: $w={w:.1f}$", fontsize=20)

    plt.tight_layout()
    plt.show()




#prserve the model.
