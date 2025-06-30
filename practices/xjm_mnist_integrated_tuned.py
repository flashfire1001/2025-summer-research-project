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
from torchvision import transforms

# --- Constants ---
IMAGE_SIZE = 28
IMAGE_CHANNELS = 1
FLAT_IMAGE_DIM = IMAGE_SIZE * IMAGE_SIZE  # 784

# --- Device Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- 1. Dataset and DataLoader ---
torch.manual_seed(42)

transform_train_cl = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.15 * torch.randn_like(x)),  # noise
    transforms.Normalize((0.5,), (0.5,))  # standardize to mean ~0
])

transform_train_fm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x))  # Tiny noise
])

train_data_fm = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform_train_fm
)

train_data_cl = datasets.MNIST(
    root="data",
    train=True,
    download=True,  # Set to True for initial download
    transform=transform_train_cl,
    target_transform=None,
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,  # Set to True for initial download
    transform=ToTensor(),
    target_transform=None,
)

BATCH_SIZE_FM = 64  # Batch size for Flow Matching training
BATCH_SIZE_CL = 32  # Batch size for Classifier training

# DataLoader for Flow Matching model
train_dataloader_fm = DataLoader(
    train_data_fm,
    batch_size=BATCH_SIZE_FM,
    shuffle=True
)


# DataLoaders for Classifier model
train_dataloader_cl = DataLoader(
    train_data_cl,
    batch_size=BATCH_SIZE_CL,
    shuffle=True
)

test_dataloader_cl = DataLoader(
    test_data,
    batch_size=BATCH_SIZE_CL,
    shuffle=False
)

# --- 2. Sample and Distribution Classes (Flow Matching) ---
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
        return X


class SampleGuassian(Sample):
    """Samples from a standard Gaussian distribution."""
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        self.image_shape = (IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

    def sample(self):
        """Returns a batch of Gaussian noise tensors."""
        X = torch.randn(self.batch_size, *self.image_shape)
        return X


# --- 3. ODE and Simulator Classes (Flow Matching ) ---
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
        pass

    def conditional_path(self, X_init: torch.Tensor, Z: torch.Tensor, t: torch.Tensor):
        """
        Calculates X(t) = (1 - t) * X_init + t * Z.
        X_init: (bs, C, H, W)
        Z: (bs, C, H, W)
        t: (bs, 1) or (bs,)
        Returns: (bs, C, H, W)
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t = t.view(t.shape[0], 1, 1, 1)

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
        return Z - X_init


class LinearPathModel(ODE):
    """
    Wraps the trained neural network (VectorFieldPredictor) to act as an ODE.
    Its 'vector_field' method will be the prediction from the network.
    """
    def __init__(self, net: nn.Module):
        self.net = net

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
        X = self.p_simple().to(device)
        ts = torch.linspace(start=0, end=1, steps=steps).to(device)
        dt = 1.0 / steps

        with torch.no_grad(): # No gradients needed for unguided inference
            for t_val in ts:
                t_for_model = t_val.repeat(X.shape[0], 1) # t_for_mode : (bs , 1)
                v_flat = self.ode.vector_field(X, t_for_model)
                v = v_flat.view(X.shape)
                X += v * dt
        return X


# --- 4. VectorFieldPredictor (Network) and Training Functions ( Flow Matching ) ---
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
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, image_dim)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x: batch of images (bs, C, H, W)
        t: batch of time values (bs, 1) or scalar t_val (0-1)
        Returns: predicted vector field (bs, FLAT_IMAGE_DIM)
        """
        x_flat = x.view(x.shape[0], -1)

        if t.dim() == 0: #if t is a scalar
            t = t.unsqueeze(0).repeat(x_flat.shape[0], 1)
        elif t.dim() > 1 and t.shape[1] > 1:
            t = t.view(t.shape[0], 1)

        x_t_cat = torch.cat((x_flat, t), dim=1)
        return self.net(x_t_cat)


def train_step_fm(ode_ref: LinearPathRef, model: VectorFieldPredictor,
               optimizer: torch.optim.Optimizer, p_simple: SampleGuassian,
               p_data: SampleData, device: str = 'cpu'):
    """
    Performs a single training epoch for the Flow Matching model.
    """
    model.train()
    total_loss = 0
    num_batches = len(p_data.dataloader)

    for _ in tqdm(range(num_batches), desc="Training FM"):

        X_init = p_simple().to(device)
        Z = p_data().to(device)
        if Z.shape[0] < X_init.shape[0]: # Adjust batch size for last incomplete batch
            X_init = X_init[:Z.shape[0]]

        t = torch.rand((X_init.shape[0], 1), device=device)

        Xt = ode_ref.conditional_path(X_init, Z, t)

        u_ref = ode_ref.conditional_vector_field(X_init, Z).view(X_init.shape[0], -1)
        u_pred = model(Xt, t)

        optimizer.zero_grad()
        loss = F.mse_loss(u_pred, u_ref)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


# --- Classifier Model and Training Functions (from Classifier code) ---
class MNISTClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 10, out_channels = 10, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels = 10, out_channels = 10, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 10, out_channels = 10, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.outputlayer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 10 * 7 * 7, out_features = 10)
        )

    def forward(self,x):
        processed_features = self.block2(self.block1(x))
        return self.outputlayer(processed_features)

def accuracy_fn(y_pred,y_truth):
    correct = torch.eq(y_pred,y_truth).sum().item()
    total = y_truth.size(0)
    return correct / total * 100

def train_step_cl(data_loader:DataLoader, model, optimizer, device, loss_fn):
    model = model.to(device)
    total_loss = 0
    total_acc = 0
    model.train()

    for i, (train_features_batch, train_labels_batch) in enumerate(tqdm(data_loader, desc="Training CL")):
        train_features_batch = train_features_batch.to(device)
        train_labels_batch = train_labels_batch.to(device)

        y_logits = model(train_features_batch)
        y_prob = torch.softmax(y_logits,dim = 1)
        y_pred = y_prob.argmax(dim = 1)

        acc = accuracy_fn(y_pred, train_labels_batch)

        optimizer.zero_grad()
        loss = loss_fn(y_logits, train_labels_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc+= acc

    train_loss = total_loss / len(data_loader)
    train_acc = total_acc /len(data_loader)
    print(f"Train loss:{train_loss:.5f} | Train_accuracy:{train_acc:.3f}%")
    return train_loss, train_acc


def test_step_cl(data_loader:DataLoader, model, device, loss_fn):
    model = model.to(device)
    total_loss = 0
    total_acc = 0
    model.eval()

    with torch.inference_mode():
        for i, (test_features_batch, test_labels_batch) in enumerate(tqdm(data_loader, desc="Testing CL")):
            test_features_batch = test_features_batch.to(device)
            test_labels_batch = test_labels_batch.to(device)

            y_logits = model(test_features_batch)
            y_prob = torch.softmax(y_logits,dim = 1)
            y_pred = y_prob.argmax(dim = 1)

            acc = accuracy_fn(y_pred, test_labels_batch)
            loss = loss_fn(y_logits, test_labels_batch)

            total_loss += loss.item()
            total_acc += acc

    test_loss = total_loss / len(data_loader)
    test_acc = total_acc /len(data_loader)
    print(f"Test loss:{test_loss:.5f} | Test accuracy:{test_acc:.3f}%")
    return test_loss, test_acc


def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())
    return torch.stack(pred_probs) # return a stack of prob, each contains 10 probabilities for every class


# --- Training Both Models ---

# Train Flow Matching Model
print("\n--- Training Flow Matching Model ---")
model_FM = VectorFieldPredictor(image_dim=FLAT_IMAGE_DIM).to(device)
optimizer_fm = torch.optim.Adam(params=model_FM.parameters(), lr=0.001)
p_simple_dist = SampleGuassian(BATCH_SIZE_FM)
linear_path_ref = LinearPathRef() # Define this before use
p_data_dist_fm = SampleData(train_dataloader_fm) # Using FM dataloader

EPOCHS_FM = 200
for epoch in range(EPOCHS_FM):
    print(f"\n--- FM Epoch {epoch + 1}/{EPOCHS_FM} ---")
    epoch_train_loss = train_step_fm(
        ode_ref=linear_path_ref,
        model=model_FM,
        optimizer=optimizer_fm,
        p_simple=p_simple_dist,
        p_data=p_data_dist_fm,
        device=device
    )
    print(f"FM Epoch {epoch + 1} average train loss: {epoch_train_loss:.5f}")


# Train Classifier Model
print("\n--- Training Classifier Model ---")
model_CL = MNISTClassifierModel().to(device)
loss_fn_cl = nn.CrossEntropyLoss()
optimizer_cl = torch.optim.SGD(params=model_CL.parameters(), lr=0.01)

EPOCHS_CL = 10 # Can increase if needed for better performance, but 5 is usually good for MNIST
for epoch in range(EPOCHS_CL):
    print(f"\n--- CL Epoch {epoch} ----")
    train_step_cl(train_dataloader_cl, model_CL, optimizer_cl, device, loss_fn_cl)
    test_step_cl(test_dataloader_cl, model_CL, device, loss_fn_cl)


# --- 5. Guided Generation Simulator ---

class Simulator_guided(Simulator):
    """
    Simulates the flow from a simple distribution (e.g., Gaussian noise)
    to a guided data distribution using the learned vector field and classifier guidance.
    """
    def __init__(self, ode: ODE, p_simple: SampleGuassian, classifier: nn.Module, target_digit: int, guidance_scale: float):
        self.ode = ode
        self.p_simple = p_simple
        self.classifier = classifier.eval() # Classifier in eval mode for inference
        self.target_digit = target_digit
        self.guidance_scale = guidance_scale

    def simulate(self, steps: int, device: str = 'cpu'):
        X = self.p_simple().to(device)
        X.requires_grad_(True) # Ensure initial X requires gradients

        ts = torch.linspace(start=0, end=1, steps=steps).to(device)
        dt = 1.0 / steps

        for t_val in ts:

            t_for_model = t_val.repeat(X.shape[0], 1)

            # --- 1. Get predicted vector field from Flow Matching model ---
            v_flow_flat = self.ode.vector_field(X, t_for_model)

            # --- 2. Calculate classifier guidance gradient ---
            logits = self.classifier(X) # X must require grad here, and this builds the graph
            log_probs = F.log_softmax(logits, dim=1)
            target_log_prob = log_probs[:, self.target_digit]
            #Use log-prob to get smoother gradients
            guidance_grad = torch.autograd.grad(outputs=target_log_prob.sum(), inputs=X, retain_graph=True)[0]

            # Compute gradients of target logit sum w.r.t. X
            #guidance_grad = torch.autograd.grad(outputs=target_logit.sum(), inputs=X, retain_graph=True)[0]

            guidance_grad_flat = guidance_grad.view(X.shape[0], -1)

            # --- 3. Combine vector fields ---

            # early steps should correct the trajectory more, while late steps fine-tune
            v_guided_flat = v_flow_flat + self.guidance_scale * guidance_grad_flat * (1 - t_for_model) ** 2

            v_guided = v_guided_flat.view(X.shape)

            # Update X for the next iteration.
            X = X + v_guided * dt


        return X.detach() # Detach final output for plotting/further use


# --- 6. Perform Guided Generation ---
print("\n--- Performing Guided Generation ---")

# Ensure models are in evaluation mode
model_FM.eval()
model_CL.eval()

# Wrap the trained Flow Matching model to be used as an ODE for simulation
trained_ode_for_sim = LinearPathModel(model_FM)

# We want to generate single images for plotting, so p_init for simulator is batch_size=1
p_init_for_guided_sim = SampleGuassian(batch_size=1)

# --- Set your target digit and guidance strength here! ---
target_digit_to_generate = 7 # Example: try to generate a '7'
guidance_strength = 2.0 # This is a crucial hyperparameter to tune (e.g., 0.5 to 5.0)

guided_simulator = Simulator_guided(
    ode=trained_ode_for_sim,
    p_simple=p_init_for_guided_sim,
    classifier=model_CL,
    target_digit=target_digit_to_generate,
    guidance_scale=guidance_strength
)

# Generate a few guided images and display them
num_samples_to_generate_guided = 3
fig_guided, axes_guided = plt.subplots(1, num_samples_to_generate_guided, figsize=(num_samples_to_generate_guided * 3, 3))
fig_guided.suptitle(f"Guided Generated MNIST Digits (Target: {target_digit_to_generate}, Scale: {guidance_strength})")



for i in range(num_samples_to_generate_guided):
    # The simulate method handles its own gradient tracking for X
    generated_image_guided = guided_simulator.simulate(steps=256, device=device)

    plot_image_guided = generated_image_guided.squeeze().cpu().numpy()
    img = plot_image_guided
    plot_image_guided = (img - img.min()) / (img.max() - img.min()) # normalize to [0,1]
    if num_samples_to_generate_guided == 1:
        axes_guided.imshow(plot_image_guided, cmap='gray')
        axes_guided.axis('off')
    else:
        axes_guided[i].imshow(plot_image_guided, cmap='gray')
        axes_guided[i].axis('off')

plt.tight_layout()
plt.show()

print("\n--- Verifying Guided Generations with Classifier ---")
test_images_for_classification = []
for _ in range(10): # Generate 10 images for verification
    test_images_for_classification.append(guided_simulator.simulate(steps=256, device=device).squeeze(0)) # Squeeze batch dim

# Make predictions on the generated images
predictions_probs = make_predictions(model_CL, test_images_for_classification, device)
predicted_labels = predictions_probs.argmax(dim=1)

print(f"Generated images predicted as: {predicted_labels.tolist()}")

# Visualize the verified images
fig_verify, axes_verify = plt.subplots(2, 5, figsize=(15, 6))
fig_verify.suptitle(f"Verification of Guided Generations (Target: {target_digit_to_generate})")
for i, ax in enumerate(axes_verify.flatten()):
    if i < len(test_images_for_classification):
        test_img = test_images_for_classification[i].squeeze().cpu().numpy()
        test_img = (test_img - test_img.min()) / (test_img.max() - test_img.min())
        ax.imshow(test_img, cmap='gray')
        ax.set_title(f"Pred: {predicted_labels[i].item()}")
        ax.axis('off')
plt.tight_layout()
plt.show()

print("Guided generation complete. Displaying results.")


from pathlib import Path

def save_model(model):
    # 1. Create models directory
    PATH = Path.cwd()
    NAME = 'flowmodel_mnist.pth'
    SAVE_PATH = PATH / 'models' / NAME
    # 2. Create model save path
    print(f"Saving model to {SAVE_PATH}")
    # 3. Save the model state dict
    torch.save(model.state_dict(), SAVE_PATH)

save_model(model_FM)
