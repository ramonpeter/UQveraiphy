import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


class VBLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        prior_prec=1.0,
        _map=False,
        std_init=-9,
        bayesian_bias=False,
    ):
        super().__init__()
        self.n_in = in_features
        self.n_out = out_features
        self.map = _map
        self.prior_prec = prior_prec
        self.random = None
        self.lbound = -30 if torch.get_default_dtype() == torch.float64 else -20
        self.eps = 1e-12 if torch.get_default_dtype() == torch.float64 else 1e-8
        self.bayesian_bias = bayesian_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        if self.bayesian_bias:
            self.bias_logsig2 = nn.Parameter(torch.Tensor(out_features))
        self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.logsig2_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.std_init = std_init
        self.reset_parameters()

    def enable_map(self):
        self.map = True

    def disable_map(self):
        self.map = False

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.mu_w.size(1))
        self.mu_w.data.normal_(0, stdv)
        self.logsig2_w.data.zero_().normal_(self.std_init, 0.001)
        self.bias.data.zero_()

    def reset_random(self):
        self.random = None
        self.map = False

    def sample_random_state(self):
        return torch.randn_like(self.logsig2_w).detach().cpu().numpy()

    def import_random_state(self, state):
        self.random = torch.tensor(
            state, device=self.logsig2_w.device, dtype=self.logsig2_w.dtype
        )

    def KL(self):
        logsig2_w = self.logsig2_w.clamp(self.lbound, 11)
        kl = (
            0.5
            * (
                self.prior_prec * (self.mu_w.pow(2) + logsig2_w.exp())
                - logsig2_w
                - 1
                - math.log(self.prior_prec)
            ).sum()
        )
        if self.bayesian_bias:
            logsig2_b = self.bias_logsig2.clamp(-30, 11)
            kl += (
                0.5
                * (
                    self.prior_prec * (self.bias.pow(2) + logsig2_b.exp())
                    - logsig2_b
                    - 1
                    - math.log(self.prior_prec)
                ).sum()
            )
        return kl

    def forward(self, input):
        if self.training:
            # local reparameterization trick is more efficient and leads to
            # an estimate of the gradient with smaller variance.
            # https://arxiv.org/pdf/1506.02557.pdf
            if self.bayesian_bias:
                bias = self.bias
                bias_logsig2 = self.bias_logsig2.clamp(self.lbound, 11)
                bias_var = bias_logsig2.exp()
                bias = bias + bias_var.sqrt() * torch.randn_like(bias)
            else:
                bias = self.bias
            mu_out = F.linear(input, self.mu_w, bias)
            logsig2_w = self.logsig2_w.clamp(self.lbound, 11)
            s2_w = logsig2_w.exp()
            var_out = F.linear(input.pow(2), s2_w) + self.eps
            # Needed to avoid NaNs from gradient of sqrt in next line!
            return mu_out + var_out.sqrt() * torch.randn_like(mu_out)

        else:
            if self.bayesian_bias:
                bias = self.bias
                bias_logsig2 = self.bias_logsig2.clamp(self.lbound, 11)
                bias_var = bias_logsig2.exp()
                bias = bias + bias_var.sqrt() * torch.randn_like(bias)
            else:
                bias = self.bias
            if self.map:
                return F.linear(input, self.mu_w, self.bias)

            logsig2_w = self.logsig2_w.clamp(self.lbound, 11)
            if self.random is None:
                self.random = torch.randn_like(self.mu_w)
            s2_w = logsig2_w.exp()
            weight = self.mu_w + s2_w.sqrt() * self.random
            return F.linear(input, weight, self.bias)

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.n_in}) -> ({self.n_out})"

    @contextmanager
    def map_mode(self):
        """Temporarily enable MAP mode (restores original setting afterward)."""
        prev_mode = self.map
        self.map = True
        try:
            yield
        finally:
            self.map = prev_mode
