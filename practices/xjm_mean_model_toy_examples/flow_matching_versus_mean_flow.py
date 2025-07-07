# @title Imports
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.optim as optim
from torch.autograd.functional import jvp

# @title Device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print("device =",device)
cwd = Path.cwd()


# @title Distributions
class GaussianGenerator:
    def __init__(self, n_dims=2, noise_std=1.0):
        self.n_dims = n_dims
        self.noise_std = noise_std # it's really a standard Guassian

    def generate(self, num_points):
        return torch.randn(num_points, self.n_dims) * self.noise_std


class CrescentGenerator:
    def __init__(self, R=1.0, r=0.6, d=0.5):
        self.R = R  # Outer radius
        self.r = r  # Inner circle radius
        self.d = d  # Offset of inner circle

    def generate(self, num_points):
        # Calculate the area ratio to estimate required samples
        outer_area = np.pi * self.R**2
        inner_area = np.pi * self.r**2
        crescent_area = outer_area - inner_area

        # Estimate required samples with 20% buffer
        n_samples = int(num_points * (outer_area / crescent_area) * 2)
        n_samples = max(n_samples, num_points)  # Ensure we generate at least num_points

        # Generate points in the outer circle
        theta = 2 * np.pi * torch.rand(n_samples) # ensure points are evenly distributed over angle
        radius = self.R * torch.sqrt(torch.rand(n_samples)) # ensure points are evenly distributed over radius

        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)

        # Filter points that are outside the inner circle
        mask = (x - self.d)**2 + y**2 > self.r**2 # check if the points is not in the inner circle
        points = torch.stack((x[mask], y[mask]), dim=1)

        # If we didn't get enough points, recursively generate more
        while points.shape[0] < num_points:
            additional_points = self.generate(num_points - points.shape[0])
            points = torch.cat((points, additional_points), dim=0)

        return points[:num_points].to(dtype=torch.float32)


class SpiralGenerator:
    def __init__(self, noise_std=0.1, n_turns=4, radius_scale=0.5):
        self.noise_std = noise_std
        self.n_turns = n_turns
        self.radius_scale = radius_scale

    def generate(self, num_points):
        max_angle = 2 * np.pi * self.n_turns
        t = torch.linspace(0, max_angle, num_points)
        t = t * torch.pow(torch.rand(num_points), 0.5)

        r = self.radius_scale * (t / max_angle + 0.1)
        x = r * torch.cos(t)
        y = r * torch.sin(t)

        x += torch.randn(num_points) * self.noise_std
        y += torch.randn(num_points) * self.noise_std
        return torch.stack([x, y], dim=1)

class CheckerboardGenerator:
    def __init__(self, grid_size=3, scale=5.0, device='cpu'):
        self.grid_size = grid_size
        self.scale = scale
        self.device = device

    def generate(self, num_points):
        grid_length = 2 * self.scale / self.grid_size
        samples = torch.zeros(0, 2).to(self.device)

        while samples.shape[0] < num_points:
            new_samples = (torch.rand(3*num_points, 2).to(self.device) - 0.5) * 2 * self.scale
            x_mask = torch.floor((new_samples[:, 0] + self.scale) / grid_length) % 2 == 0
            y_mask = torch.floor((new_samples[:, 1] + self.scale) / grid_length) % 2 == 0
            accept_mask = torch.logical_xor(~x_mask, y_mask)
            samples = torch.cat([samples, new_samples[accept_mask]], dim=0)

        return samples[:num_points]

# @title Example Display
# Initialize generators with different parameters
generators = [
    GaussianGenerator(n_dims=2, noise_std=0.5),
    CrescentGenerator(R=1.0, r=0.6, d=0.5),
    SpiralGenerator(noise_std=0.05, n_turns=3, radius_scale=2),
    CheckerboardGenerator(grid_size=4, scale=1.5)
]

# # check if I generated correctly
#
# # Generate samples from each distribution
# num_points = 4000
# samples = [generator.generate(num_points) for generator in generators]
# titles = ["Gaussian", "Crescent", "Spiral", "Checkerboard"]
#
# # Plot the distributions
# fig, axes = plt.subplots(1, 4, figsize=(20, 5))
# for ax, sample, title in zip(axes, samples, titles):
#     ax.scatter(sample[:, 0], sample[:, 1], s=1, alpha=0.2)
#     ax.set_title(title, fontsize = 20)
#     ax.set_xlim(-2, 2)
#     ax.set_ylim(-2, 2)
#     ax.tick_params(axis='both', which='major', labelsize=20)
#     ax.set_aspect('equal')
# plt.tight_layout()
# plt.show()

# @title Visualization Functions
def visualize_denoising_progress(
    n_epochs,
    initial_points,
    num_steps_multi,
    model_class,
    model_kwargs,
    target_points,
    integration_fn,
    checkpoint_prefix='mean_flow_model',
    epoch_step=10,
    device='cpu',
    suptitle="Model Performance Over Time"
):
    """
    Visualizes denoising results across training epochs using two schemes:
    multi-step and single-step denoising.

    Args:
        n_epochs: Total number of training epochs.
        initial_points: Noisy input points (torch.Tensor).
        num_steps_multi: Integration steps for multi-step denoising.
        model_class: Class of the flow model (e.g., MeanFlowNet).
        model_kwargs: Dict of kwargs to initialize model (e.g., {'input_dim': 2, 'h_dim': 128}).
        target_points: Target distribution (numpy array).
        integration_fn: Function that performs forward integration.
        checkpoint_prefix: Prefix of checkpoint files.
        epoch_step: Interval at which to visualize checkpoints.
        device: Torch device.
        suptitle: Title for the entire figure.
    """
    steps = list(range(0, n_epochs + 1, epoch_step))
    n_rows = len(steps)
    fig, axes = plt.subplots(n_rows, 2, figsize=(8, 4 * n_rows))

    for i, epoch in enumerate(steps):
        model_name = f'{checkpoint_prefix}_epoch_{epoch}.pt'
        model_path = Path.cwd() / 'models' / model_name
        print(f"Loading model {model_name}")

        # Instantiate and load model
        plot_model = model_class(**model_kwargs).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        plot_model.load_state_dict(checkpoint['model_state_dict'])
        plot_model.eval()

        # Multi-step denoising
        denoised_multi = integration_fn(
            initial_points,
            model=plot_model,
            t_start=1.0, #we start at t = 1, corresponding to noise
            t_end=0.0, #and end at t = 0, corresponding to data
            num_steps=num_steps_multi,
            save_trajectory=False,
            device = device
        ).cpu().numpy()

        # Single-step denoising
        denoised_single = integration_fn(
            initial_points,
            model=plot_model,
            t_start=1.0,
            t_end=0.0,
            num_steps=1,
            save_trajectory=False,
            device = device
        ).cpu().numpy()

        # Left: multi-step
        ax_left = axes[i, 0]
        ax_left.scatter(target_points[:, 0], target_points[:, 1], s=5, alpha=0.05, color='blue', label='Target')
        ax_left.scatter(denoised_multi[:, 0], denoised_multi[:, 1], s=5, alpha=0.3, color='green', label='Denoised')
        ax_left.set_xlim(-3, 3)
        ax_left.set_ylim(-3, 3)
        ax_left.set_aspect('equal')
        if i == 0:
            ax_left.set_title(f'Denoising Steps = {num_steps_multi}', fontsize=20)
        ax_left.set_ylabel(f"Epoch {epoch}", fontsize=16)
        ax_left.grid(True, alpha=0.3)

        # Right: single-step
        ax_right = axes[i, 1]
        ax_right.scatter(target_points[:, 0], target_points[:, 1], s=5, alpha=0.05, color='blue', label='Target')
        ax_right.scatter(denoised_single[:, 0], denoised_single[:, 1], s=5, alpha=0.3, color='green', label='Denoised')
        ax_right.set_xlim(-3, 3)
        ax_right.set_ylim(-3, 3)
        ax_right.set_aspect('equal')
        if i == 0:
            ax_right.set_title('Denoising Steps = 1', fontsize=20)
        ax_right.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(suptitle, fontsize=25, y=1.02)
    plt.show()

# @title Mean Flow Related Code

# Mean Flow Model
class MeanFlowNet(nn.Module):
    def __init__(self, input_dim, h_dim=64):
        super().__init__()
        # Input dimension should be x (input_dim) + t (1) + r (1) = input_dim + 2
        self.fc_in  = nn.Linear(input_dim + 2, h_dim) # in this case imput_dim = 2
        self.fc2    = nn.Linear(h_dim, h_dim)
        self.fc3    = nn.Linear(h_dim, h_dim)
        self.fc4    = nn.Linear(h_dim, h_dim)
        self.fc_out = nn.Linear(h_dim, input_dim)

    def forward(self, x, t, r, act=nn.ReLU()):
        #or F.gelu
        # The GELU nonlinearity weights inputs by their percentile, rather than gates inputs by their sign as in ReLUs  Consequently the GELU can be thought of as a smoother ReLU.

        t = t.expand(x.size(0), 1)  # Ensure t has the correct dimensions for x batches
        r = r.expand(x.size(0), 1)  # Add r for meanflow! (bs, 1)

        x = torch.cat([x, t, r], dim=1)
        x = act(self.fc_in(x))
        x = act(self.fc2(x)) # always wrap it with a activation function
        x = act(self.fc3(x))
        x = act(self.fc4(x))
        return self.fc_out(x)

# MeanFlow class that handles time generation, loss computation
class MeanFlow:
    def __init__(self,):
        super().__init__()

    def sample_t_r(self, batch_size, device):
        # Generate random t values in the shape of the batch size
        samples = torch.rand(batch_size, 2, device=device)

        # Assign the smaller values to r, larger values to t, unsqueeze to make it fit the 2D data
        t = torch.max(samples[:, 0], samples[:, 1]).unsqueeze(1) # shape(bs,1)
        # torch.max(a,b) is a element wise maximum and a.max() is the maximum of all elements in a
        # give dim in this case to use .max()
        #t = samples.max(dim = 1, keepdim = True)
        r = torch.min(samples[:, 0], samples[:, 1]).unsqueeze(1)
        return t, r

    def loss(self, model, target_samples, source_samples):
        batch_size = target_samples.shape[0]
        device = target_samples.device

        t, r = self.sample_t_r(batch_size, device) # Generate t, r

        interpolated_samples = (1 - t) * target_samples + t * source_samples # x_t
        velocity = source_samples - target_samples # velocity takes targets to sources, find the conditional instantaneous velocity

        ## Mean Flow Specific Loss Calculation ##
        jvp_args = (model,(interpolated_samples, t, r),(velocity, torch.ones_like(t), torch.zeros_like(r)), )
        u, dudt = jvp(*jvp_args, create_graph=True)
        #create_graph=True 会保留雅可比向量积计算过程中的中间计算图，从而支持对 JVP 结果继续求导（高阶导数)/ 之后再反向传播.
        #jvp(func, inputs , v) compute the dot product of the jacobian matrix of the func at point inputs with the vector v. the create_graph parameter should be false: stop gradient! no more further gradient on the du/dt ! avoid double gradient.
        # 否! 这里必须要是True 因为, u是要被之后求梯度的
        # return func output and the dot product

        u_tgt = velocity - (t - r) * dudt

        ## NOTE: Very important to use .detach().这里du/dt是不在被求梯度了
        ## This sort of gradient based loss is very unstable otherwise
        ## and models can get stuck in extremely terrible local minima!!

        loss = F.mse_loss(u, u_tgt.detach()) ## Default MSE Loss
        return loss

# Training function that uses the class
def train_mean_model(model, source_data_function, target_data_function, n_epochs=100, lr=0.003, batch_size=2048, batches_per_epoch=10, epoch_save_freq = 10, checkpoint_prefix='mean_flow_model'):
    optimizer = optim.Adam(model.parameters(), lr=lr)#torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0) #different optimizer
    device = next(model.parameters()).device
    #define an instance of meanflow to use to handle times and loss calculation

    meanflow = MeanFlow()

    for epoch in range(n_epochs):
        # train for 100 epochs
        model.train()
        total_loss = 0.0
        for batch_idx in range(batches_per_epoch):
            #10 batches in a epoch
            # obtain points
            source_samples = source_data_function(batch_size).to(device)
            target_samples = target_data_function(batch_size).to(device)

            # Use points in the meanflow class
            loss = meanflow.loss(model, target_samples, source_samples)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            total_loss += loss.item()
        avg_loss = total_loss / batches_per_epoch
        print(f"Epoch [{epoch}/{n_epochs}], Avg Loss: {avg_loss:.4f}")

        if epoch % epoch_save_freq == 0:
            # Save model checkpoint
            checkpoint_path = cwd / 'models' /f'{checkpoint_prefix}_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")



    # Always save final model at the end
    checkpoint_path = cwd / 'models' /f'{checkpoint_prefix}_epoch_{n_epochs}.pt'
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, checkpoint_path)
    print(f"Saved model checkpoint to {checkpoint_path}")
    return model





# @title Flow Matching Model + Training Function
class VelocityNet(nn.Module):
    def __init__(self, input_dim, h_dim=64):
        super().__init__()
        self.fc_in  = nn.Linear(input_dim + 1, h_dim)
        self.fc2    = nn.Linear(h_dim, h_dim)
        self.fc3    = nn.Linear(h_dim, h_dim)
        self.fc_out = nn.Linear(h_dim, input_dim)

    def forward(self, x, t, act=F.gelu):
        t = t.expand(x.size(0), 1)  # Ensure t has the correct dimensions
        x = torch.cat([x, t], dim=1)
        x = act(self.fc_in(x))
        x = act(self.fc2(x))
        x = act(self.fc3(x))
        return self.fc_out(x)

def train_model(model, source_data_function, target_data_function, n_epochs=100, lr=0.003, batch_size=2048, batches_per_epoch=10, epoch_save_freq = 10, checkpoint_prefix='flow_model'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    device = next(model.parameters()).device
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for batch_idx in range(batches_per_epoch):
            optimizer.zero_grad()

            # obtain points
            source_samples = source_data_function(batch_size).to(device)
            target_samples = target_data_function(batch_size).to(device)

            t = torch.rand(source_samples.size(0), 1).to(device)  # random times for traning
            interpolated_samples = (1 - t) * target_samples + t * source_samples
            velocity = source_samples - target_samples # velocity takes targets to sources

            velocity_prediction = model(interpolated_samples, t)
            loss = loss_fn(velocity_prediction, velocity)

            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        avg_loss = total_loss / batches_per_epoch
        print(f"Epoch [{epoch+1}/{n_epochs}], Avg Loss: {avg_loss:.4f}")

        if epoch % epoch_save_freq == 0:
            # Save model checkpoint
            checkpoint_path = cwd / 'models' /f'{checkpoint_prefix}_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")

    # Always save final model at the end
    checkpoint_path = cwd / 'models' /f'{checkpoint_prefix}_epoch_{n_epochs}.pt'
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, checkpoint_path)
    print(f"Saved model checkpoint to {checkpoint_path}")
    return model
# @title Numerical Integration - Flow Matching
def forward_euler_integration_model(
    initial_points: torch.Tensor,
    model: nn.Module,
    t_start: float = 1.0,
    t_end: float = 0.0,
    num_steps: int = 100,
    save_trajectory: bool = True,
    device = 'cpu'
) -> torch.Tensor:

    dt = (t_start - t_end) / num_steps
    trajectory = [initial_points.clone()] if save_trajectory else None
    current_points = initial_points.clone().to(device)

    for step in range(0, num_steps):
        current_time = t_start - step*dt
        t_tensor = torch.full((len(current_points), 1), current_time,
                            device=current_points.device)

        with torch.no_grad():
            velocity = model(current_points, t_tensor)

        current_points += -velocity * dt # velocity takes us from targets to source

        if save_trajectory:
            trajectory.append(current_points.clone())

    return torch.stack(trajectory) if save_trajectory else current_points


# @title Numerical Integration - Mean Flow
def forward_euler_integration_mean_model(
    initial_points: torch.Tensor,
    model: nn.Module,
    t_start: float = 1.0,
    t_end: float = 0.0,
    num_steps: int = 100,
    save_trajectory: bool = True,
    device = 'cpu'
) -> torch.Tensor:

    dt = (t_start - t_end) / num_steps
    trajectory = [initial_points.clone()] if save_trajectory else None
    current_points = initial_points.clone().to(device)

    for step in range(0, num_steps):
        current_t = t_start - step*dt
        current_r = current_t - dt # Remember, the r values are smaller than t!
        #print ("Current time ", current_r , " moving to ", current_t)
        t_tensor = torch.full((len(current_points), 1), current_t,
                            device=current_points.device)
        r_tensor = torch.full((len(current_points), 1), current_r,
                            device=current_points.device)

        with torch.no_grad():
            velocity = model(current_points, t_tensor, r_tensor)

        current_points += -velocity * dt  # negative because our velocity moves from target to source

        if save_trajectory:
            trajectory.append(current_points.clone())

    return torch.stack(trajectory) if save_trajectory else current_points

# experiment1 --- compare the effectiveness: checkerboard ---

# Initialize distributions
source_generator = GaussianGenerator(n_dims=2, noise_std=1.0)  # Mean=0, Std=1
target_generator = CheckerboardGenerator(grid_size=4, scale=1.5)

# Generate test data
num_points = 4000
initial_points = source_generator.generate(num_points)
target_points = target_generator.generate(num_points)



model = VelocityNet(input_dim=2, h_dim=128).to(device)
checkpoint_prefix='flow_model'

n_epoch_flow = 60

start_time = time.time()

trained_model = train_model(
    model=model,
    source_data_function=source_generator.generate,
    target_data_function=target_generator.generate,
    n_epochs=n_epoch_flow,
    lr=0.003,
    batch_size=4096,
    batches_per_epoch=50,
    epoch_save_freq=10,
    checkpoint_prefix=checkpoint_prefix
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training took {elapsed_time:.2f} seconds")


visualize_denoising_progress(
    n_epochs=n_epoch_flow,
    initial_points=initial_points,
    num_steps_multi=100,
    model_class=VelocityNet,
    model_kwargs={'input_dim': 2, 'h_dim': 128},  # model initialization parameters
    target_points=target_points,
    integration_fn = forward_euler_integration_model,
    checkpoint_prefix=checkpoint_prefix,
    suptitle="Normal Flow Model",
    epoch_step=20,
    device=device
)

mean_model = MeanFlowNet(input_dim=2, h_dim=128).to(device)
mean_checkpoint_prefix='mean_flow_model'

# Train the model
n_epochs = 100

start_time = time.time()

trained_model = train_mean_model(
    model=mean_model,
    source_data_function=source_generator.generate,
    target_data_function=target_generator.generate,
    n_epochs=n_epochs,
    lr=0.003,
    batch_size=4096,
    batches_per_epoch=50,
    epoch_save_freq=10,
    checkpoint_prefix=mean_checkpoint_prefix
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training took {elapsed_time:.2f} seconds")


visualize_denoising_progress(
    n_epochs=n_epochs,
    initial_points=initial_points,
    num_steps_multi=10,
    model_class=MeanFlowNet,
    model_kwargs={'input_dim': 2, 'h_dim': 128},  # model initialization parameters
    target_points=target_points,
    integration_fn = forward_euler_integration_mean_model,
    checkpoint_prefix=mean_checkpoint_prefix,
    suptitle="Mean Flow Model",
    epoch_step=20,
    device=device
)



# experiment2 --- compare the effectiveness: crescent ---

# @title Initialize Crescent
target_generator_crescent = CrescentGenerator(R=1.0, r=0.6, d=0.5)

# Generate test data
num_points = 4000
initial_points = source_generator.generate(num_points)
target_points = target_generator_crescent.generate(num_points)

# @title Flow Matching Model
model = VelocityNet(input_dim=2, h_dim=128).to(device)
crescent_checkpoint_prefix='crescent_flow_model'

# @title Training
n_epoch_flow = 60

start_time = time.time()

trained_model = train_model(
    model=model,
    source_data_function=source_generator.generate,
    target_data_function=target_generator_crescent.generate,
    n_epochs=n_epoch_flow,
    lr=0.003,
    batch_size=4096,
    batches_per_epoch=50,
    epoch_save_freq=10,
    checkpoint_prefix=crescent_checkpoint_prefix
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training took {elapsed_time:.2f} seconds")

# @title Visualization
visualize_denoising_progress(
    n_epochs=n_epoch_flow,
    initial_points=initial_points,
    num_steps_multi=100,
    model_class=VelocityNet,
    model_kwargs={'input_dim': 2, 'h_dim': 128},  # model initialization parameters
    target_points=target_points,
    integration_fn = forward_euler_integration_model,
    checkpoint_prefix=crescent_checkpoint_prefix,
    suptitle="Normal Flow Model",
    epoch_step=20,
    device=device
)

# @title Mean Flow Model
model = MeanFlowNet(input_dim=2, h_dim=128).to(device)
mean_crescent_checkpoint_prefix='crescent_mean_flow_model'

# @title Mean Flow Training
n_epochs = 100

start_time = time.time()

trained_model = train_mean_model(
    model=model,
    source_data_function=source_generator.generate,
    target_data_function=target_generator_crescent.generate,
    n_epochs=n_epochs,
    lr=0.003,
    batch_size=4096,
    batches_per_epoch=50,
    epoch_save_freq=10,
    checkpoint_prefix=mean_crescent_checkpoint_prefix
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training took {elapsed_time:.2f} seconds")

# @title Mean Flow Visualization
visualize_denoising_progress(
    n_epochs=n_epochs,
    initial_points=initial_points,
    num_steps_multi=100,
    model_class=MeanFlowNet,
    model_kwargs={'input_dim': 2, 'h_dim': 128},  # model initialization parameters
    target_points=target_points,
    integration_fn = forward_euler_integration_mean_model,
    checkpoint_prefix=mean_crescent_checkpoint_prefix,
    suptitle="Mean Flow Model",
    epoch_step=20,
    device=device
)


# experiment3 --- compare the effectiveness: Spiral- the hardest for two model! ---


# @title Initialize Spiral
target_generator_spiral = SpiralGenerator(noise_std=0.05, n_turns=3, radius_scale=2)

# Generate test data
num_points = 4000
initial_points = source_generator.generate(num_points)
target_points = target_generator_spiral.generate(num_points)

# @title Flow Matching Model
model = VelocityNet(input_dim=2, h_dim=128).to(device)
spiral_checkpoint_prefix='spiral_flow_model'

# @title Training
n_epoch_flow = 80

start_time = time.time()

trained_model = train_model(
    model=model,
    source_data_function=source_generator.generate,
    target_data_function=target_generator_spiral.generate,
    n_epochs=n_epoch_flow,
    lr=0.003,
    batch_size=4096,
    batches_per_epoch=50,
    epoch_save_freq=10,
    checkpoint_prefix=spiral_checkpoint_prefix
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training took {elapsed_time:.2f} seconds")

# @title Visualization
visualize_denoising_progress(
    n_epochs=n_epoch_flow,
    initial_points=initial_points,
    num_steps_multi=100,
    model_class=VelocityNet,
    model_kwargs={'input_dim': 2, 'h_dim': 128},  # model initialization parameters
    target_points=target_points,
    integration_fn = forward_euler_integration_model,
    checkpoint_prefix=spiral_checkpoint_prefix,
    suptitle="Normal Flow Model",
    epoch_step=20,
    device=device
)

# @title Mean Flow Model
model = MeanFlowNet(input_dim=2, h_dim=128).to(device)
mean_spiral_checkpoint_prefix='spiral_mean_flow_model'

# @title Mean Flow Training
n_epochs = 500

start_time = time.time()

trained_model = train_mean_model(
    model=model,
    source_data_function=source_generator.generate,
    target_data_function=target_generator_spiral.generate,
    n_epochs=n_epochs,
    lr=0.003,
    batch_size=4096,
    batches_per_epoch=50,
    epoch_save_freq=100,
    checkpoint_prefix=mean_spiral_checkpoint_prefix
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training took {elapsed_time:.2f} seconds")

# @title Mean Flow Visualization
visualize_denoising_progress(
    n_epochs=n_epochs,
    initial_points=initial_points,
    num_steps_multi=100,
    model_class=MeanFlowNet,
    model_kwargs={'input_dim': 2, 'h_dim': 128},  # model initialization parameters
    target_points=target_points,
    integration_fn = forward_euler_integration_mean_model,
    checkpoint_prefix=mean_spiral_checkpoint_prefix,
    suptitle="Mean Flow Model",
    epoch_step=100,
    device=device
)

