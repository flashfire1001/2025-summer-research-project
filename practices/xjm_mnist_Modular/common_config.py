# ============ config_common.py ============
import torch

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MiB = 1024**2

def model_size_b(model):
    size = sum(p.nelement() * p.element_size() for p in model.parameters())
    size += sum(b.nelement() * b.element_size() for b in model.buffers())
    return size