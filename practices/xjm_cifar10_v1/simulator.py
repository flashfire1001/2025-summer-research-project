from abc import ABC, abstractmethod
import torch
from tqdm import tqdm

from ode_path import ODE

# --- part3 : simulator ---
class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs):
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            # you only have to go for n - 1 steps
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        xs = [x.clone()]
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)

# for tensor shape operations
'''
y = x is just for creating a new reference, not copying any data. 
Hence:
Both x and y point to the same memory.
Changes to y affect x, and vice versa
y = x.clone()
deep copy: create a tensor with its own memory contain the same value
use it when you want to :
modify y but keep the x unchanged
safe for backprop
slower

x.view or x.reshape() 
x and y share the same underlying data.
y = x.view Only works if the reshaping is compatible with how the data is laid out in memory (i.e., it's contiguous).

x.reshape() is safer; and fast as x.view() mostly
it tries to clone if needed

'''

class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        return xt + self.ode.drift_coefficient(xt, t, **kwargs) * h

