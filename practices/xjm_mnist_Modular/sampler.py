

from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import torch
import torch.nn as nn
from torchvision import datasets, transforms

class Sampleable(ABC):
    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """sample both a optional label and a data tensor"""
        pass


class IsotropicGaussian(nn.Module, Sampleable):
    def __init__(self, shape: List[int], std: float = 1.0):
        super().__init__()
        self.shape = shape
        self.std = std
        self.dummy = nn.Buffer(torch.zeros(1))

    def sample(self, num_samples) -> Tuple[torch.Tensor, torch.Tensor]:
        """return a tensor of size (num_samples, *shape), with no label."""
        return self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device), None


class MNISTSampler(nn.Module, Sampleable):
    def __init__(self):
        super().__init__()
        self.dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                #very common and completely normal in practice, especially when adapting it to models
                # originally designed for other datasets like CIFAR-10
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)), #normals the element(pixel value) in the image to be in (-1,1) for faster computation
            ])
        )
        self.dummy = nn.Buffer(torch.zeros(1))

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.dataset)}")

        indices = torch.randperm(len(self.dataset))[:num_samples]
        samples, labels = zip(*[self.dataset[i] for i in indices])
        samples = torch.stack(samples).to(
            self.dummy)
        labels = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)
        return samples, labels


