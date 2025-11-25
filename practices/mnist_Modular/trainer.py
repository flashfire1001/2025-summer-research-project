# ============ trainer.py ============
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from tqdm import tqdm
from sampler import Sampleable
from model import ConditionalVectorField
from ode_path import GaussianConditionalProbabilityPath

from common_config import device, model_size_b

class Trainer(ABC):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs):
        loss_record = [] # track the loss fluctuation during training .
        size_b = model_size_b(self.model)
        print(f'Training model with size: {size_b / (1024**2):.2f} MiB')
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            loss_record.append(loss.cpu().item())
        self.model.eval()
        return loss_record

class CFGTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, model: ConditionalVectorField, eta: float):
        #assert 0.0 < eta < 1.0
        super().__init__(model)
        self.eta = eta
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z, y = self.path.p_data.sample(batch_size)
        z, y = z.to(device), y.to(device)

        mask = torch.rand(y.shape[0], device=device) < self.eta
        y[mask] = 10  # classifier-free token

        t = torch.rand(batch_size, 1, 1, 1, device=device)
        x_t = self.path.sample_conditional_path(z, t)

        target = self.path.conditional_vector_field(x_t, z, t)
        pred = self.model(x_t, t, y)

        return torch.mean(torch.square(pred - target))
