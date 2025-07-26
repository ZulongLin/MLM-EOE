from mamba_ssm import Mamba
from torch import nn


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba_ssm = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

    def forward(self, x):
        x_norm = self.norm(x)
        x_mamba = self.mamba_ssm(x_norm)
        return x_mamba
