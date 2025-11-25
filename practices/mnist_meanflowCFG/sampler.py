

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


class TimeSampler(nn.Module, Sampleable):
    def __init__(self, equal_rate:float =0.5):
        super().__init__()
        self.dummy = nn.Buffer(torch.zeros(1))
        self.equal_rate = equal_rate

    def sample(self, num_samples, distribution: str = None, mu=-0.4, std=1):

        # separate into equal and random cases
        equal_num = int(self.equal_rate * num_samples)

        if distribution == "uniform":
            # create full random samples
            samples = torch.rand((num_samples, 2), device=self.dummy.device)

            #rand_num = self.num_samples - equal_num

            #  For equal samples, set t = r = one value
            t_equal = torch.rand(equal_num, device=self.dummy.device)
            r_equal = t_equal.clone()

            #  For others, set t = max, r = min as before
            samples_rand = samples[equal_num:]
            t_rand = torch.max(samples_rand[:, 0], samples_rand[:, 1])
            r_rand = torch.min(samples_rand[:, 0], samples_rand[:, 1])

        else:
            # logit normal by default

            z = torch.randn(num_samples).to(self.dummy.device)
            z = (z + mu) * std
            z = torch.sigmoid(z)
            t_equal = z[:equal_num]
            r_equal = t_equal.clone()


            z_rand = z[equal_num:]
            t_rand = torch.max((1 - z_rand), z_rand)
            r_rand = torch.min((1 - z_rand), z_rand)

        #  Concatenate and shuffle
        t_all = torch.cat([t_equal, t_rand], dim=0)
        r_all = torch.cat([r_equal, r_rand], dim=0)

        #  shuffle so equal/random aren't clustered
        idx = torch.randperm(num_samples)
        t_all = t_all[idx]
        r_all = r_all[idx]

        return t_all.reshape(num_samples,1,1,1), r_all.reshape(num_samples,1,1,1)



