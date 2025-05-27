"""
This code is based on https://github.com/tatsy/normalizing-flows-pytorch
"""
import numpy as np
import torch
import torch.nn as nn
from modeling.Flow_based.modules import LipSwish


def safe_detach(x):
    """
    detech operation which keeps reguires_grad
    ---
    https://github.com/rtqichen/residual-flows/blob/master/lib/layers/iresblock.py
    """
    return x.detach().requires_grad_(x.requires_grad)


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class ResFlow(nn.Module):
    def __init__(self, dims, n_layers):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(InvertibleResLinear(dims[0], dims[0]))

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


class SpectralNorm(nn.Module):
    """
    modified spectral normalization [Miyato et al. 2018] for invertible residual networks
    ---
    most of this implementation is borrowed from the following link:
    https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """

    def __init__(self, module, coeff=0.97, eps=1.0e-5, name="weight", power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.coeff = coeff
        self.eps = eps
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        scale = self.coeff / (sigma + self.eps)

        delattr(self.module, self.name)
        if scale < 1.0:
            setattr(self.module, self.name, w * scale.expand_as(w))
        else:
            setattr(self.module, self.name, w)

    def _made_params(self):
        try:
            _ = getattr(self.module, self.name + "_u")
            _ = getattr(self.module, self.name + "_v")
            _ = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = w.data.new(height).normal_(0, 1)
        v = w.data.new(width).normal_(0, 1)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        self.module.register_buffer(self.name + "_u", u)
        self.module.register_buffer(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def log_df_dz_unbias(g, z, n_samples=1, p=0.5, n_exact=1, is_training=True):
    """
    log determinant approximation using unbiased series length sampling
    which can be used with residual block f(x) = x + g(x)
    """

    res = 0.0
    for j in range(n_samples):
        n_power_series = n_exact + np.random.geometric(p)

        v = torch.randn_like(g)
        w = v

        sum_vj = 0.0
        for k in range(1, n_power_series + 1):
            w = torch.autograd.grad(g, z, w, create_graph=is_training, retain_graph=True)[0]
            geom_cdf = (1.0 - p) ** max(0, (k - n_exact) - 1)
            tr = torch.sum(w * v, dim=1)
            sum_vj = sum_vj + (-1) ** (k + 1) * (tr / (k * geom_cdf))

        res += sum_vj

    return res / n_samples


def log_df_dz_neumann(g, z, n_samples=1, p=0.5, n_exact=1):
    """
    log determinant approximation using unbiased series length sampling.
    ---
    NOTE: this method using neumann series does not return exact "log_df_dz"
    but the one that can be only used in gradient wrt parameters.
    """

    res = 0.0
    for j in range(n_samples):
        n_power_series = n_exact + np.random.geometric(p)

        v = torch.randn_like(g)
        w = v

        sum_vj = v
        with torch.no_grad():
            for k in range(1, n_power_series + 1):
                w = torch.autograd.grad(g, z, w, retain_graph=True)[0]
                geom_cdf = (1.0 - p) ** max(0, (k - n_exact) - 1)
                sum_vj = sum_vj + ((-1) ** k / geom_cdf) * w

        sum_vj = torch.autograd.grad(g, z, sum_vj, create_graph=True)[0]
        res += torch.sum(sum_vj * v, dim=1)

    return res / n_samples


class MemorySavedLogDetEstimator(torch.autograd.Function):
    """
    Memory saving logdet estimator used in Residual Flow
    ---
    This code is borrowed from following URL but revised as it can be understood more easily.
    https://github.com/rtqichen/residual-flows/blob/master/lib/layers/iresblock.py
    """

    @staticmethod
    def forward(ctx, logdet_fn, x, g_fn, training, *g_params):
        ctx.training = training
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            theta = list(g_params)

            # log-det for neumann series
            g = g_fn(x)
            ctx.x = x
            ctx.g = g
            logdetJg = log_df_dz_neumann(g, x)

            if ctx.training:
                dlogdetJg_dx, *dlogdetJg_dtheta = torch.autograd.grad(logdetJg.sum(), [x] + theta, retain_graph=True, allow_unused=True)
                ctx.save_for_backward(dlogdetJg_dx, *theta, *dlogdetJg_dtheta)

            # log-det for loss calculation
            logdet = logdet_fn(g, x)

        return safe_detach(g), safe_detach(logdet)

    @staticmethod
    def backward(ctx, dL_dg, dL_dlogdetJg):
        """
        NOTE: Be careful that chain rule for partial differentiation is as follows

        df(y, z)    df   dy     df   dz
        -------- =  -- * --  +  -- * --
        dx          dy   dx     dz   dx
        """

        training = ctx.training
        if not training:
            raise ValueError("Provide training=True if using backward.")

        # chain rule for partial differentiation (1st term)
        with torch.enable_grad():
            g, x = ctx.g, ctx.x
            dlogdetJg_dx, *saved_tensors = ctx.saved_tensors
            n_params = len(saved_tensors) // 2
            theta = saved_tensors[:n_params]
            dlogdetJg_dtheta = saved_tensors[n_params:]

            dL_dx_1st, *dL_dtheta_1st = torch.autograd.grad(g, [x] + theta, grad_outputs=dL_dg, allow_unused=True)

        # chain rule for partial differentiation (2nd term)
        # ---
        # NOTE:
        # dL_dlogdetJg consists of same values for all dimensions (see forward).
        dL_dlogdetJg_scalar = dL_dlogdetJg[0].detach()
        with torch.no_grad():
            dL_dx_2nd = dlogdetJg_dx * dL_dlogdetJg_scalar
            dL_dtheta_2nd = tuple([g * dL_dlogdetJg_scalar if g is not None else None for g in dlogdetJg_dtheta])

        with torch.no_grad():
            dL_dx = dL_dx_1st + dL_dx_2nd
            dL_dtheta = tuple([g1 + g2 if g2 is not None else g1 for g1, g2 in zip(dL_dtheta_1st, dL_dtheta_2nd)])

        return (None, dL_dx, None, None) + dL_dtheta


def memory_saved_logdet_wrapper(logdet_fn, x, g_fn, training):
    g_params = list(g_fn.parameters())
    return MemorySavedLogDetEstimator.apply(logdet_fn, x, g_fn, training, *g_params)


class InvertibleResLinear(nn.Module):
    def __init__(self, in_chs, out_chs, mid_chs=32, n_layers=2, coeff=0.97, ftol=1.0e-4):
        super().__init__()

        self.coeff = coeff
        self.ftol = ftol
        self.proc_g_fn = memory_saved_logdet_wrapper

        act_fn = LipSwish
        hidden_dims = [in_chs] + [mid_chs] * n_layers + [out_chs]
        layers = nn.ModuleList()
        for i, (in_dims, out_dims) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            layers.append(SpectralNorm(nn.Linear(in_dims, out_dims), coeff=self.coeff))
            if i != len(hidden_dims) - 2:
                layers.append(act_fn())

        self.g_fn = nn.Sequential(*layers)

    def _get_logdet_estimator(self):
        if self.training:
            logdet_fn = lambda g, z: log_df_dz_unbias(g, z, 1, is_training=True)
        else:
            logdet_fn = lambda g, z: log_df_dz_unbias(g, z, n_samples=4, n_exact=8, is_training=False)
        return logdet_fn

    def forward(self, x, log_df_dz):
        logdet_fn = self._get_logdet_estimator()
        g, logdet = self.proc_g_fn(logdet_fn, x, self.g_fn, self.training)
        z = x + g
        log_df_dz += logdet
        return z, log_df_dz

    def backward(self, z, log_df_dz):
        n_iters = 100
        x = z.clone()
        logdet_fn = self._get_logdet_estimator()

        with torch.enable_grad():
            x.requires_grad_(True)
            for k in range(n_iters):
                x = safe_detach(x)
                g = self.g_fn(x)
                x, prev_x = z - g, x

                if torch.all(torch.abs(x - prev_x) < self.ftol):
                    break

            x = safe_detach(x)
            g = self.g_fn(x)
            logdet = logdet_fn(g, x)

        return x, log_df_dz - logdet
