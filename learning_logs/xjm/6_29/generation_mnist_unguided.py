import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# --- Constants ---
IMAGE_SIZE = 28
IMAGE_CHANNELS = 1
FLAT_IMAGE_DIM = IMAGE_SIZE * IMAGE_SIZE  # 784

# --- 1. Dataset and DataLoader ---
torch.manual_seed(42)

# Ensure download=True if you haven't downloaded it before
train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,  # Set to True for initial download
    transform=ToTensor(),
    target_transform=None,
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,  # Set to True for initial download
    transform=ToTensor(),
    target_transform=None,
)

BATCH_SIZE = 64  # Increased batch size for potentially better performance

train_dataloader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_dataloader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --- 2. Sample and Distribution Classes ---
class Sample(ABC):
    """Abstract base class for sampling distributions."""

    @abstractmethod
    def sample(self):
        pass

    def __call__(self):
        return self.sample()


class SampleData(Sample):
    """Samples a single batch of data from a DataLoader."""

    def __init__(self, dataloader: DataLoader):
        super().__init__()
        self.dataloader = dataloader
        self.data_iter = iter(self.dataloader)  # Initialize iterator

    def sample(self):
        """Returns a single batch of image tensors."""
        try:
            X, _ = next(self.data_iter)
        except StopIteration:
            # Reached end of epoch, re-initialize iterator
            self.data_iter = iter(self.dataloader)
            X, _ = next(self.data_iter)
        return X  # Shape: (batch_size, C, H, W)


class SampleGuassian(Sample):
    """Samples from a standard Gaussian distribution."""

    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        self.image_shape = (IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

    def sample(self):
        """Returns a batch of Gaussian noise tensors."""
        X = torch.randn(self.batch_size, *self.image_shape)
        return X  # Shape: (batch_size, C, H, W)


# --- 3. ODE and Simulator ---
class ODE(ABC):
    """Abstract base class for Ordinary Differential Equations."""

    @abstractmethod
    def conditional_path(self, X_init, Z, t):
        """Defines the path X(t) between X_init and Z."""
        pass

    @abstractmethod
    def vector_field(self, X, t):
        """Defines the vector field u(X, t) = dX/dt."""
        pass


class LinearPathRef(ODE):
    """
    Defines a simple linear path and its corresponding vector field
    for conditional flow matching. This serves as the 'ground truth'
    for the model to learn.
    """

    def __init__(self):
        pass  # No need for p_simple/p_data here, they are passed externally

    def conditional_path(self, X_init: torch.Tensor, Z: torch.Tensor, t: torch.Tensor):
        """
        Calculates X(t) = (1 - t) * X_init + t * Z.
        X_init: (bs, C, H, W)
        Z: (bs, C, H, W)
        t: (bs, 1) or (bs,)
        Returns: (bs, C, H, W)
        """
        # Ensure t is (bs, 1, 1, 1) for broadcasting across image dimensions
        if t.dim() == 1:
            t = t.unsqueeze(1)  # (bs,) -> (bs, 1)
        t = t.view(t.shape[0], 1, 1, 1)  # (bs, 1) -> (bs, 1, 1, 1)

        Xt = (1 - t) * X_init + t * Z
        return Xt

    def vector_field(self, X: torch.Tensor, t: torch.Tensor):
        """
        Placeholder for a marginal vector field. Not directly used for training
        in this conditional flow matching setup, but could be for other ODEs.
        """
        raise NotImplementedError("This method is not used in this specific Flow Matching setup.")

    def conditional_vector_field(self, X_init: torch.Tensor, Z: torch.Tensor):
        """
        Returns the vector field for the linear path: u(x_t|x_0, x_1) = Z - X_init.
        This is the target for our neural network.
        """
        return Z - X_init  # Shape: (bs, C, H, W)


class LinearPathModel(ODE):
    """
    Wraps the trained neural network (VectorFieldPredictor) to act as an ODE.
    Its 'vector_field' method will be the prediction from the network.
    """

    def __init__(self, net: nn.Module):
        self.net = net
        # LinearPathModel should inherit from nn.Module if it has learnable params,
        # but here it's just a wrapper, so no super().__init__() is needed for nn.Module

    def conditional_path(self, X_init, Z, t):
        """Conditional path not directly used by this wrapper for training."""
        raise NotImplementedError("This model doesn't define a conditional path; it predicts the vector field.")

    def vector_field(self, X: torch.Tensor, t: torch.Tensor):
        """
        Returns the vector field predicted by the neural network.
        X: (bs, C, H, W)
        t: (bs, 1)
        Returns: (bs, FLAT_IMAGE_DIM)
        """
        # The network expects (bs, FLAT_IMAGE_DIM) for X and (bs, 1) for t
        # It returns (bs, FLAT_IMAGE_DIM)
        return self.net(X, t)


class Simulator(ABC):
    """Abstract base class for simulating ODEs."""

    @abstractmethod
    def simulate(self, steps: int, device: str = 'cpu'):
        """Simulates the process of flow / numerical integration."""
        pass


class Simulator_unguided(Simulator):
    """
    Simulates the flow from a simple distribution (e.g., Gaussian noise)
    to the data distribution using the learned vector field.
    """

    def __init__(self, ode: ODE, p_simple: SampleGuassian):
        self.ode = ode  # This will be an instance of LinearPathModel
        self.p_simple = p_simple  # This will be SampleGuassian

    def simulate(self, steps: int, device: str = 'cpu'):
        """
        Performs Euler integration to generate samples.
        X_0 is sampled from p_simple.
        """
        # X starts as (batch_size, C, H, W) from the simple distribution
        X = self.p_simple().to(device)
        ts = torch.linspace(start=0, end=1, steps=steps).to(device)
        dt = 1.0 / steps

        for t_val in ts:
            # Create a batch of 't' for the current time step.
            # It needs to be (batch_size, 1) to match the network's expected input.
            t_for_model = t_val.repeat(X.shape[0], 1)

            # Get the predicted vector field at current X and t_val
            # self.ode.vector_field calls model_FM(X, t_for_model)
            # which returns (batch_size, FLAT_IMAGE_DIM)
            v_flat = self.ode.vector_field(X, t_for_model)

            # Reshape the vector field back to (batch_size, C, H, W)
            v = v_flat.view(X.shape)

            # Euler step: X_next = X_current + v * dt
            X += v * dt

        return X


# --- 4. VectorFieldPredictor (Network) and Training Functions ---
class VectorFieldPredictor(nn.Module):
    """
    Neural network to predict the vector field for Flow Matching.
    Input: (flattened_image_data, t)
    Output: predicted_vector_field (same shape as flattened_image_data)
    """

    def __init__(self, image_dim: int = FLAT_IMAGE_DIM):
        super().__init__()
        self.image_dim = image_dim
        self.net = nn.Sequential(
            nn.Linear(image_dim + 1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),  # Added layer
            nn.ReLU(),
            nn.Linear(512, 512),  # Added layer
            nn.ReLU(),
            nn.Linear(512, image_dim)  # Output is the predicted vector field
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x: batch of images (bs, C, H, W)
        t: batch of time values (bs, 1) or scalar t_val (0-1)
        Returns: predicted vector field (bs, FLAT_IMAGE_DIM)
        """
        # Flatten image data: (bs, C, H, W) -> (bs, FLAT_IMAGE_DIM)
        x_flat = x.view(x.shape[0], -1)

        # Ensure t is (bs, 1). Handle if t is a scalar (e.g., during simulation)
        if t.dim() == 0:  # If t is a single scalar (e.g., 0.5)
            t = t.unsqueeze(0).repeat(x_flat.shape[0], 1)  # Expand to (bs, 1)
        elif t.dim() > 1 and t.shape[1] > 1:  # If t is (bs, 1, 1, 1) from LinearPathRef
            t = t.view(t.shape[0], 1)  # Reshape to (bs, 1)

        # Concatenate flattened image and time (bs, FLAT_IMAGE_DIM + 1)
        x_t_cat = torch.cat((x_flat, t), dim=1)

        return self.net(x_t_cat)


def train_step(ode_ref: LinearPathRef, model: VectorFieldPredictor,
               optimizer: torch.optim.Optimizer, p_simple: SampleGuassian,
               p_data: SampleData, device: str = 'cpu'):
    """
    Performs a single training epoch for the Flow Matching model.
    """
    model.train()  # Set model to training mode
    total_loss = 0
    num_batches = len(p_data.dataloader)

    # Use tqdm for a progress bar
    for _ in tqdm(range(num_batches), desc="Training",leave=False):
        # Sample X_0 from the simple distribution and X_1 (Z) from the data distribution
        X_init = p_simple().to(device)  # (bs, C, H, W)
        Z = p_data().to(device)  # (bs, C, H, W)
        if Z.shape[0] < 64 :
            Z = torch.cat((Z,Z),dim = 0)

        #print(f"Shape of X_init: {X_init.shape}")
        #print(f"Shape of Z: {Z.shape}")
        # Sample time 't' uniformly for each sample in the batch
        t = torch.rand((X_init.shape[0], 1), device=device)  # (bs, 1)

        # Get the interpolated point Xt = (1-t)X_init + tZ
        Xt = ode_ref.conditional_path(X_init, Z, t)

        # Get the reference vector field u_ref = Z - X_init
        # Flatten u_ref to (bs, FLAT_IMAGE_DIM) to match model output
        u_ref = ode_ref.conditional_vector_field(X_init, Z).view(X_init.shape[0], -1)

        # Get the predicted vector field u_pred from the model
        u_pred = model(Xt, t)  # Model outputs (bs, FLAT_IMAGE_DIM)

        # Zero gradients, compute loss, backpropagate, and update weights
        optimizer.zero_grad()
        loss = F.mse_loss(u_pred, u_ref)  # MSE loss (mean reduction by default)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # .item() to get Python scalar from tensor

    avg_loss = total_loss / num_batches
    return avg_loss


# --- Set up Training Fundamentals ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Create an instance of the VectorFieldPredictor (our neural network)
model_FM = VectorFieldPredictor(image_dim=FLAT_IMAGE_DIM).to(device)

# Adam optimizer with a standard learning rate
optimizer = torch.optim.Adam(params=model_FM.parameters(), lr=0.001)

# Instances of our sampling classes
p_simple_dist = SampleGuassian(BATCH_SIZE)
p_data_dist = SampleData(train_dataloader)

# Instance of our linear path reference ODE
linear_path_ref = LinearPathRef()

EPOCHS = 50  # Increased number of epochs

print(f"Starting training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
    epoch_train_loss = train_step(
        ode_ref=linear_path_ref,
        model=model_FM,
        optimizer=optimizer,
        p_simple=p_simple_dist,
        p_data=p_data_dist,
        device=device
    )

    print(f"Epoch {epoch + 1} average train loss: {epoch_train_loss:.5f}")

# --- 5. Evaluate Model (Generate Samples) ---
print("\n--- Generating samples ---")

# Set model to evaluation mode
model_FM.eval()

# To generate, we need an ODE that uses our trained model's predictions
trained_ode_for_sim = LinearPathModel(model_FM)

# We want to generate single images for plotting, so p_init for simulator is batch_size=1
p_init_for_sim = SampleGuassian(batch_size=1)

# Simulator instance for generation
simulator = Simulator_unguided(trained_ode_for_sim, p_init_for_sim)

# Generate a few images and display them
num_samples_to_generate = 5
fig, axes = plt.subplots(1, num_samples_to_generate, figsize=(num_samples_to_generate * 3, 3))

with torch.no_grad():  # Disable gradient calculation for inference
    for i in range(num_samples_to_generate):
        # Simulate the flow to generate an image
        # Use a reasonable number of steps for simulation (e.g., 100 or 256)
        generated_image = simulator.simulate(steps=256, device=device)

        # Reshape for plotting: (1, C, H, W) -> (H, W)
        plot_image = generated_image.squeeze().cpu().numpy()

        if num_samples_to_generate == 1:
            axes.imshow(plot_image, cmap='gray')
            axes.axis('off')
        else:
            axes[i].imshow(plot_image, cmap='gray')
            axes[i].axis('off')

plt.suptitle(f"Generated MNIST Digits after {EPOCHS} Epochs")
plt.tight_layout()
plt.show()

