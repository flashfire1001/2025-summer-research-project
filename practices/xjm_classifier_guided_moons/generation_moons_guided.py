from typing import Optional, List, Type, Tuple, Dict
import numpy as np
from matplotlib import pyplot as plt
import torch.distributions as D
from sklearn.datasets import make_moons
from pathlib import Path as Pth
import torch
from torch import nn

#device agnostic
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def duplicate_model(loaded_model,save_path):
    # Create new instance of model and load saved state dict (make sure to put it on the target device)
    loaded_model.load_state_dict(torch.load(save_path))
    loaded_model.to(device)
    return loaded_model


# Inherit from nn.Module to make a model capable of fitting the mooon data
class MoonModelV0(nn.Module):
    ## Your code here ##
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(2,10)
        self.layer_2 = nn.Linear(10,10)
        self.layer_3 = nn.Linear(10,10)
        self.layer_4 = nn.Linear(10,2)
        self.relu = nn.ReLU()

    def forward(self, x):
        ## Your code here ##
        return self.relu(self.layer_4(self.relu(self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x))))))))


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
        self.p_simple = SampleGaussian.isotropic(dim=2, std=1.0)
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

    def simulate_with_guidance(self, num_samples, classifer, label = 0):
        ts = torch.linspace(0.0, 1.0, self.steps).to(device)
        dt = 1.0 / self.steps  # CORRECTED: Use dt based on self.steps
        x = self.path.p_simple.sample(num_samples).to(device)  # This is x_0
        x.requires_grad_(True)  # Enable gradient tracking

        for i_t in range(self.steps):
            t = ts[i_t].item() * torch.ones((x.shape[0], 1)).to(device)
            logits = classifer(x)  # shape: (batch_size, num_classes)
            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            log_p_label = log_probs[:, label]  # select log p(y|x) for the target class y
            # Compute gradient of log p(label|x) w.r.t. x
            grad = torch.autograd.grad(log_p_label.sum(), x)[0]  # shape: same as x
            x = x + self.vector_field.drift_coefficient(x, t) * dt + t*t * grad * 0.5
        return x




# load the models
PATH = Pth.cwd()
NAME = 'generation_moons_unguided.pth'
SAVE_PATH = PATH / 'models' / NAME
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_FM = FlowModel().to(device)
model_FM = duplicate_model(model_FM,SAVE_PATH)
NAME = 'classification_moons.pth'
SAVE_PATH = PATH / 'models' / NAME
model_CL = MoonModelV0()
model_CL = duplicate_model(model_CL,SAVE_PATH)

path = Path(SampleMoons())
velocity_field = LearnedVectorFieldODE(model_FM)
simulator = Simulator(velocity_field,10,path) # 100 steps for simulation

# now I only want to craft for one part/side of the moons
img = simulator.simulate_with_guidance(2000,model_CL).cpu().detach().numpy() # 1000 samples for simulation output
x,y = (img[:,0] ,img[:,1])
fig,ax = plt.subplots(figsize = (7,7))
ax.scatter(x,y,s = 2)
ax.set_title("Generated Samples from Learned Flow")
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
plt.show() # Display the plot


