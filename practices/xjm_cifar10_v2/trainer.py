# ============ trainer.py ============
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from tqdm import tqdm
from model import ConditionalVectorField
from ode_path import GaussianConditionalProbabilityPath
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ChainedScheduler
from common_config import device, model_size_b
from ema import EMA

class Trainer(ABC):
    def __init__(self, model: nn.Module,ema_decay :float =0.999):
        super().__init__()
        self.model = model
        self.ema = EMA(model, decay = ema_decay)

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs):
        size_b = model_size_b(self.model)
        print(f'Training model with size: {size_b / (1024**2):.2f} MiB')
        loss_record = []
        self.model.to(device)
        opt = self.get_optimizer(lr)

        # Warmup scheduler: linearly increase lr from 0 to 1 over warmup_steps=500
        def warmup_lambda(step):
            return min(1.0, step / 500)

        warmup_scheduler = LambdaLR(opt, lr_lambda=warmup_lambda)

        # Cosine annealing scheduler over entire num_epochs
        cosine_scheduler = CosineAnnealingLR(opt, T_max=num_epochs)

        # Chain schedulers: warmup first, then cosine annealing
        scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])

        self.model.train()
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            self.ema.update()
            pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            loss_record.append(loss.cpu().item())
            scheduler.step()
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
