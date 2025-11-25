# ============ config_common.py ============
import torch
from pathlib import Path


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MiB = 1024**2

def model_size_b(model):
    size = sum(p.nelement() * p.element_size() for p in model.parameters())
    size += sum(b.nelement() * b.element_size() for b in model.buffers())
    return size


def save_model(epoch, model, guidance_scale, loss = None):
    cwd = Path.cwd()
    checkpoint_path = cwd / 'models' / f'guidance_scale{guidance_scale}_epoch_{epoch}.pt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, checkpoint_path)


def load_model(epoch,guidance_scale,model_class,**model_kwargs,):
    cwd = Path.cwd()
    model_name = f'guidance_scale{guidance_scale}_epoch_{epoch}.pt'
    model_path = cwd / 'models' / model_name
    print(f"Loading model {model_name}")

    # Instantiate and load model
    plot_model = model_class(**model_kwargs).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    plot_model.load_state_dict(checkpoint['model_state_dict'])
    plot_model.eval()
    return plot_model
