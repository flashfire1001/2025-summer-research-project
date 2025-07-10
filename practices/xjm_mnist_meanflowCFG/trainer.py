# ============ trainer.py ============
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from tqdm import tqdm
from sampler import Sampleable
from model import ConditionalVectorField
from ode_path import GaussianConditionalProbabilityPath
from torch.autograd.functional import jvp
import torch.nn.functional as F


from common_config import device, model_size_b, save_model

class Trainer(ABC):
    def __init__(self, model: nn.Module, guidance_scale):
        super().__init__()
        self.model = model
        self.guidance_scale = guidance_scale

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
            #if epoch % 500 ==0 :
                #save_model(epoch, self.model, guidance_scale= self.guidance_scale,loss = loss.detach().item())
        save_model(num_epochs, self.model, guidance_scale=self.guidance_scale) #for temporary test
        #print(f"Saved model after going through all epochs")
        self.model.eval()
        return loss_record

class CFGTrainer(Trainer):
    def __init__(self,
                 path: GaussianConditionalProbabilityPath,
                 model: ConditionalVectorField, eta: float,
                 guidance_scale:float,
                 t_sampler:Sampleable):
        #assert 0.0 < eta < 1.0
        super().__init__(model,guidance_scale=guidance_scale)
        self.eta = eta
        self.path = path
        self.t_sampler = t_sampler


    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z, y = self.path.p_data.sample(batch_size)
        z, y = z.to(device), y.to(device)

        mask = torch.rand(y.shape[0], device=device) < self.eta
        y[mask] = 10  # classifier-free token

        #sample t and r
        t,r = self.t_sampler.sample(batch_size)
        

        #sample x_t
        x_t = self.path.sample_conditional_path(z, t)

        v_instant_cond = self.path.conditional_vector_field(x_t, z, t)

        v_instant_marg = self.model(x_t, t, t, ) #y=torch.tensor([10]).to(t.device)

        v_cfg = self.guidance_scale* v_instant_cond + (1-self.guidance_scale)*v_instant_marg

        jvp_args = (self.model, (x_t, t, r), (v_cfg, torch.ones_like(t), torch.zeros_like(r)),)
        u_cfg, dudt = jvp(*jvp_args, create_graph=True)

        u_tgt = v_cfg - (t - r) * dudt

        loss = F.mse_loss(u_cfg, u_tgt.detach())

        return loss


