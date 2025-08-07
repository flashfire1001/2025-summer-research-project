from typing import Optional, List, Type, Tuple, Dict
import math

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.distributions as D
from sklearn.datasets import make_moons


#device agnostic
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# we define 2 classes for sample

class SampleMoons():
    '''make a batch of moons tensor'''

    def __init__(self,  scale=4):
        self.scale = scale

    def sample(self, num_samples):
        samples , _ = make_moons(num_samples, noise = 0.02 ,random_state = 42)
        return samples * self.scale

class SampleGaussian(torch.nn.Module):
    """
    Multivariate Gaussian distribution
    """
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        """
        mean: shape (dim,)
        cov: shape (dim,dim)
        """
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    @property
    def distribution(self):
        return D.MultivariateNormal(self.mean, self.cov, validate_args=False)

    def sample(self, num_samples) -> torch.Tensor:
        return self.distribution.sample((num_samples,))


    @classmethod
    def isotropic(cls, dim: int =2, std: float =1.0) -> "SampleGaussian": # Changed return type to SampleGaussian
        mean = torch.zeros(dim)
        cov = torch.eye(dim) * std ** 2
        return cls(mean, cov)

# create the simulator and the path
class Path:
    """
    the design path tells position and velocity at any time.
    here I choose linear path for simplicity
    """

    def __init__(self,img_type):
        self.p_simple = SampleGaussian.isotropic(dim=2, std=3.0)
        self.p_data = img_type

    def sample_condition_variable(self, num_samples):
        z = torch.from_numpy(self.p_data.sample(num_samples)).type(torch.float)
        return z

    # MODIFIED: Now returns both x_t and the sampled x_init
    def conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """specific position and initial noise"""
        x_init = self.p_simple.sample(z.shape[0]).to(device).type(torch.float)
        x_t = (torch.ones_like(t)-t) * x_init + t * z
        return x_t, x_init # Return both x_t and x_init

    # MODIFIED: Corrected reference velocity field. It should be z - x_init.
    # x and t are not needed as inputs for the true linear path velocity.
    def conditional_v_field(self, z: torch.Tensor, x_init: torch.Tensor) -> torch.Tensor:
        """The true conditional velocity field for a linear path."""
        return z - x_init


class LearnedVectorFieldODE:
    def __init__(self, net):
        self.net = net

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: (bs, dim)
            - t: (bs, 1)
        Returns:
            - u_t: (bs, dim)
        """
        return self.net(x, t)

class Simulator:
    """ODE euler simulator for integration"""
    def __init__(self, ode:LearnedVectorFieldODE, steps, path:Path):
        self.vector_field = ode# the ODE
        self.steps = steps
        self.path = path

    def simulate(self,num_samples):
        ts = torch.linspace(0.0, 1.0, self.steps).to(device)
        dt = 1.0 / self.steps # CORRECTED: Use dt based on self.steps
        x = self.path.p_simple.sample(num_samples).to(device) # This is x_0
        for i_t in range(self.steps):
            t = ts[i_t].item() * torch.ones((x.shape[0],1)).to(device)
            x = x + self.vector_field.drift_coefficient(x, t) * dt # CORRECTED: Use dt here
        return x

class FlowModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 64), # Input is x (2D) + t (1D) = 3D
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,2), # Output is 2D velocity vector
        )

    def forward(self, x , t):
        x_t = torch.cat((x, t),dim = 1)
        return self.layers(x_t)

class Trainer():

    def __init__(self,model:FlowModel,path:Path,num_samples,epochs):
        self.path = path
        self.model = model
        self.num_samples =  num_samples
        self.epochs = epochs

    def get_train_loss(self):
        z = self.path.sample_condition_variable(self.num_samples).to(device)
        t = torch.rand(self.num_samples).unsqueeze(dim = 1).to(device)

        # MODIFIED: Get both x_t and x_init from conditional_path
        x, x_init = self.path.conditional_path(z, t) # x is x_t

        # MODIFIED: Calculate v_ref using z and the specific x_init for this batch
        v_ref = self.path.conditional_v_field(z, x_init).to(device)
        v_pred = self.model(x, t).to(device)
        mean_loss_batch = torch.sum(torch.square(v_ref-v_pred),dim = -1)

        # Changed num_samples to self.num_samples for consistency
        return (1.0/self.num_samples)* torch.sum(mean_loss_batch,dim = 0)

    def get_optimizer(self, lr: float):
        # This method is defined but not used. The optimizer is set up directly in train().
        return torch.optim.Adam(self.model.layers.parameters(), lr=lr)

    def train(self):
        #set up the training loop
        self.model.to(device)
        self.model.train()
        opt = torch.optim.Adam(self.model.layers.parameters(), lr=0.01)

        for epoch in range(self.epochs):
            # go forward and get the loss.
            loss = self.get_train_loss()
            # zero the optimizer
            opt.zero_grad()
            # loss backpropagation for tuning the optimizer
            loss.backward()
            # update step the parameter
            opt.step()

            if epoch % 100 == 0:
                print(f"in epoch {epoch}, the training loss: {loss.item()}") # .item() to get scalar value


num_samples = 2000 # This num_samples is for the trainer
path = Path(SampleMoons())
model = FlowModel()
trainer = Trainer(model, path, num_samples, epochs = 2500)

trainer.train()
velocity_field = LearnedVectorFieldODE(model)
simulator = Simulator(velocity_field,10,path) # 100 steps for simulation

img = simulator.simulate(5000).cpu().detach().numpy() # 1000 samples for simulation output
x,y = (img[:,0] ,img[:,1])
fig,ax = plt.subplots(figsize = (7,7))
ax.scatter(x,y,s = 2)
ax.set_title("Generated Samples from Learned Flow")
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
plt.show() # Display the plot

from pathlib import Path

def save_model(model):
    # 1. Create models directory
    PATH = Path.cwd()
    NAME = 'generation_moons_unguided.pth'
    SAVE_PATH = PATH / 'models' / NAME
    # 2. Create model save path
    print(f"Saving model to {SAVE_PATH}")
    # 3. Save the model state dict
    torch.save(model.state_dict(), SAVE_PATH)

save_model(model)