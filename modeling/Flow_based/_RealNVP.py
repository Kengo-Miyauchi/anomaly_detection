"""
This code is based on https://github.com/tatsy/normalizing-flows-pytorch
"""
import torch
import torch.nn as nn
from modeling.Flow_based.modules import ConvNet, BatchNorm, Logit
from modeling.Flow_based.squeeze import squeeze1d, unsqueeze1d, Squeeze2d, Unsqueeze2d, checker_split, checker_merge, channel_split, channel_merge


class RealNVP(nn.Module):
    def __init__(self, dims, n_layers):
        super(RealNVP, self).__init__()

        self.dims = dims

        self.layers = nn.ModuleList()
        if len(dims) == 3:
            # for image data
            self.layers.append(Logit(eps=0.01))

            # multi-scale architecture
            mid_dims = dims
            while max(mid_dims[1], mid_dims[2]) > 8:
                # checkerboard masking
                for i in range(n_layers):
                    self.layers.append(BatchNorm(mid_dims, affine=False))
                    self.layers.append(AffineCoupling(mid_dims, masking='checkerboard', odd=i % 2 != 0))

                # squeeze
                self.layers.append(Squeeze2d(odd=False))
                mid_dims = (mid_dims[0] * 4, mid_dims[1] // 2, mid_dims[2] // 2)

                # channel-wise masking
                for i in range(n_layers):
                    self.layers.append(BatchNorm(mid_dims, affine=False))
                    self.layers.append(AffineCoupling(mid_dims, masking='channelwise', odd=i % 2 != 0))

            # checkerboard masking (lowest resolution)
            for i in range(n_layers + 1):
                self.layers.append(BatchNorm(mid_dims, affine=False))
                self.layers.append(AffineCoupling(mid_dims, masking='checkerboard', odd=i % 2 != 0))

            # restore to original scale
            while mid_dims[1] != dims[1] or mid_dims[2] != dims[2]:
                # unsqueeze
                self.layers.append(Unsqueeze2d(odd=False))
                mid_dims = (mid_dims[0] // 4, mid_dims[1] * 2, mid_dims[2] * 2)

        else:
            # for density samples
            for i in range(n_layers):
                self.layers.append(BatchNorm(dims, affine=False))
                self.layers.append(AffineCoupling(dims, odd=i % 2 != 0))

    def forward(self, z):
        log_df_dz = 0
        for layer in self.layers:
            z, log_df_dz = layer(z, log_df_dz)
        return z, log_df_dz

    def backward(self, z):
        log_df_dz = 0
        for layer in reversed(self.layers):
            z, log_df_dz = layer.backward(z, log_df_dz)
        return z, log_df_dz


class ResBlockLinear(nn.Module):
    def __init__(self, in_chs, out_chs):
        super().__init__()
        self.net = nn.Sequential(
            # nn.BatchNorm1d(in_chs),
            nn.ReLU(True),
            nn.Linear(in_chs, out_chs),
            # nn.BatchNorm1d(out_chs),
            nn.ReLU(True),
            nn.Linear(out_chs, out_chs),
        )
        self.relu = nn.ReLU(True)
        if in_chs != out_chs:
            self.shorcut = nn.Sequential(
                nn.Linear(in_chs, out_chs),
            )
        else:
            self.shorcut = nn.Sequential()

    def forward(self, x):
        y = self.net(x)
        y += self.shorcut(x)
        return y


class MLP(nn.Module):
    def __init__(self, in_chs, out_chs, mid_chs=32, n_blocks=2):
        super().__init__()
        self.in_block = nn.Linear(in_chs, mid_chs)
        self.mid_blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.mid_blocks.append(ResBlockLinear(mid_chs, mid_chs))
        self.out_block = nn.Sequential(
            # nn.BatchNorm1d(mid_chs),
            nn.ReLU(True),
            nn.Linear(mid_chs, out_chs),
        )

    def forward(self, x):
        x = self.in_block(x)
        for layer in self.mid_blocks:
            x = layer(x)
        x = self.out_block(x)
        return x


class AbstractCoupling(nn.Module):
    """
    abstract class for bijective coupling layers
    """
    def __init__(self, dims, masking='checkerboard', odd=False):
        super(AbstractCoupling, self).__init__()
        self.dims = dims
        if len(dims) == 1:
            self.squeeze = lambda z, odd=odd: squeeze1d(z, odd)
            self.unsqueeze = lambda z0, z1, odd=odd: unsqueeze1d(z0, z1, odd)
        elif len(dims) == 3 and masking == 'checkerboard':
            self.squeeze = lambda z, odd=odd: checker_split(z, odd)
            self.unsqueeze = lambda z0, z1, odd=odd: checker_merge(z0, z1, odd)
        elif len(dims) == 3 and masking == 'channelwise':
            self.squeeze = lambda z, odd=odd: channel_split(z, dim=1, odd=odd)
            self.unsqueeze = lambda z0, z1, odd=odd: channel_merge(z0, z1, dim=1, odd=odd)
        else:
            raise Exception('unsupported combination of masking and dimension: %s, %s' %
                            (masking, str(dims)))

    def forward(self, z, log_df_dz):
        z0, z1 = self.squeeze(z)
        z0, z1, log_df_dz = self._transform(z0, z1, log_df_dz)
        z = self.unsqueeze(z0, z1)
        return z, log_df_dz

    def backward(self, y, log_df_dz):
        y0, y1 = self.squeeze(y)
        y0, y1, log_df_dz = self._inverse_transform(y0, y1, log_df_dz)
        y = self.unsqueeze(y0, y1)

        return y, log_df_dz

    def _transform(self, z0, z1, log_df_dz):
        pass

    def _inverse_transform(self, z0, z1, log_df_dz):
        pass


class AffineCoupling(AbstractCoupling):
    """
    affine coupling used in Real NVP
    """
    def __init__(self, dims, masking='checkerboard', odd=False):
        super(AffineCoupling, self).__init__(dims, masking, odd)

        self.register_parameter('s_log_scale', nn.Parameter(torch.randn(1) * 0.01))
        self.register_parameter('s_bias', nn.Parameter(torch.randn(1) * 0.01))

        if len(dims) == 1:
            in_chs = dims[0] // 2 if not odd else (dims[0] + 1) // 2
            self.out_chs = dims[0] - in_chs
            self.net = MLP(in_chs, self.out_chs * 2)
        elif len(dims) == 3:
            if masking == 'checkerboard':
                in_out_chs = dims[0] * 2
            elif masking == 'channelwise':
                in_out_chs = dims[0] // 2
            self.out_chs = in_out_chs
            self.net = ConvNet(in_out_chs, in_out_chs * 2)

    def _transform(self, z0, z1, log_df_dz):
        params = self.net(z1)
        t = params[:, :self.out_chs]
        s = torch.tanh(params[:, self.out_chs:]) * self.s_log_scale + self.s_bias

        z0 = z0 * torch.exp(s) + t
        log_df_dz += torch.sum(s.view(z0.size(0), -1), dim=1)

        return z0, z1, log_df_dz

    def _inverse_transform(self, y0, y1, log_df_dz):
        params = self.net(y1)
        t = params[:, :self.out_chs]
        s = torch.tanh(params[:, self.out_chs:]) * self.s_log_scale + self.s_bias

        y0 = torch.exp(-s) * (y0 - t)
        log_df_dz -= torch.sum(s.view(y0.size(0), -1), dim=1)

        return y0, y1, log_df_dz
