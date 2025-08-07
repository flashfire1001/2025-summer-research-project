import math
import torch
import torch.nn as nn

class FourierEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #assert dim % 2 == 0
        self.weights = nn.Parameter(torch.randn(1, dim // 2))

    def forward(self, t):
        t = t.view(-1, 1)
        freqs = t * self.weights * 2 * math.pi
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1) * math.sqrt(2)

class ResidualLayer(nn.Module):
    def __init__(self, channels, t_dim, y_dim):
        super().__init__()
        self.block1 = nn.Sequential(nn.SiLU(), nn.BatchNorm2d(channels), nn.Conv2d(channels, channels, 3, 1, 1))
        self.block2 = nn.Sequential(nn.SiLU(), nn.BatchNorm2d(channels), nn.Conv2d(channels, channels, 3, 1, 1))
        self.t_adapter = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, channels))
        self.y_adapter = nn.Sequential(nn.Linear(y_dim, y_dim), nn.SiLU(), nn.Linear(y_dim, channels))

    def forward(self, x, t, y):
        """why the layer is designed like this?"""
        """
        This allows the network to learn only a residual function (i.e., what's different from the input).
        
        Helps with gradient flow and avoids vanishing gradients in deep architectures.
        
        Residual connections enable the model to preserve information and only update selectively.
        """
        res = x.clone()
        x = self.block1(x)
        x = x + self.t_adapter(t).unsqueeze(-1).unsqueeze(-1)
        x = x + self.y_adapter(y).unsqueeze(-1).unsqueeze(-1)
        x = self.block2(x)
        return x + res

class Encoder(nn.Module):
    def __init__(self, cin, cout, nres, tdim, ydim):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualLayer(cin, tdim, ydim) for _ in range(nres)])
        self.downsample = nn.Conv2d(cin, cout, 3, 2, 1)

    def forward(self, x, t, y):
        for blk in self.blocks:
            x = blk(x, t, y)
        return self.downsample(x)

class Midcoder(nn.Module):
    def __init__(self, ch, nres, tdim, ydim):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualLayer(ch, tdim, ydim) for _ in range(nres)])

    def forward(self, x, t, y):
        for blk in self.blocks:
            x = blk(x, t, y)
        return x

class Decoder(nn.Module):
    def __init__(self, cin, cout, nres, tdim, ydim):
        super().__init__()
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(cin, cout, 3, 1, 1))
        self.blocks = nn.ModuleList([ResidualLayer(cout, tdim, ydim) for _ in range(nres)])

    def forward(self, x, t, y):
        x = self.upsample(x)
        for blk in self.blocks:
            x = blk(x, t, y)
        return x

class ConditionalVectorField(nn.Module):
    def forward(self, x, t, y):
        raise NotImplementedError

class CIFARUNet(ConditionalVectorField):
    def __init__(self, channels, nres, tdim, ydim):
        super().__init__()
        self.init = nn.Sequential(nn.Conv2d(3, channels[0], 3, 1, 1), nn.BatchNorm2d(channels[0]), nn.SiLU())
        self.time_embed = FourierEncoder(tdim)
        self.label_embed = nn.Embedding(11, ydim)
        self.encs = nn.ModuleList()
        self.decs = nn.ModuleList()
        # how does this works? how many encoders are on the list?
        for curr_c, next_c in zip(channels[:-1], channels[1:]):
            self.encs.append(Encoder(curr_c, next_c, nres, tdim, ydim))
            self.decs.insert(0, Decoder(next_c, curr_c, nres, tdim, ydim))
        self.mid = Midcoder(channels[-1], nres, tdim, ydim)
        self.final = nn.Conv2d(channels[0], 3, 3, 1, 1)

    def forward(self, x, t, y):
        """use x as the receiver and parameter who shares the same memory, this makes the operation faster."""
        t_embed = self.time_embed(t)
        y_embed = self.label_embed(y)
        x = self.init(x)
        skips = []
        for enc in self.encs:
            x = enc(x, t_embed, y_embed)
            skips.append(x.clone())
        x = self.mid(x, t_embed, y_embed)
        for dec in self.decs:
            #in typical U-Net architectures, the encoder skip connection is often concatenated with the decoder feature map along the channel dimension
            """
            Richer representation: The decoder gets full access to both encoder features and decoder context.
            Encourages the decoder to learn how to combine old and new features flexibly.
            require an additional convolition or projection layer
            More parameters and computation.
            """

            # the addition is simplier and parameter-free merging
            # saves memory and computation hence faster
            # Enforces strict channel match between encoder and decoder(conv does this similarly, though)
            #Less flexible — may reduce expressive capacity, especially when skip and decoder features have different meanings.
            x = x + skips.pop()
            #as we are residual based, the model learns what to change not to regenerate the whole signal.
            #Adding the skip connection encourages the network to preserve low-level information and only apply refinements.

            x = dec(x, t_embed, y_embed)
        return self.final(x)
