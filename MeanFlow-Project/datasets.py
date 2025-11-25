import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import add_gaussian_noise, add_sp_noise, add_block_mask


transform_ = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.5, ), std=(0.5, ))
])


class NoisyMNIST(torch.utils.data.Dataset):
    def __init__(self, root, noise_func, noise_func_arg, train=True, download=True):
        self.dataset = datasets.MNIST(
            root = root,
            train = train,
            download = True,
            transform = transform_
        )
        self.noise_func = noise_func
        self.noise_func_arg = noise_func_arg

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        clean_img, label = self.dataset[index]
        noised_img = self.noise_func(clean_img.clone(), self.noise_func_arg)
        return noised_img, clean_img, label


NOISE_FUNCTIONS = {
    "gaussian": add_gaussian_noise,
    "sp": add_sp_noise,
    "block": add_block_mask
}


def get_noised_mnist_dataloader(
        noise_func: str = "gaussian",
        noise_func_arg: float = 0.5,
        root: str = "data/mnist",
        batch_size: int = 128,
        num_workers: int = 4,
        train: bool = True,
        download: bool = True
) -> DataLoader:

    if isinstance(noise_func, str):
        if noise_func not in NOISE_FUNCTIONS:
            raise ValueError(f"Unknown noise_func '{noise_func}'. "
                             f"Available: {list(NOISE_FUNCTIONS.keys())}")
        noise_func = NOISE_FUNCTIONS[noise_func]

    dataset = NoisyMNIST(root, noise_func, noise_func_arg, train, download)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return dataloader
