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
        use_map=False,
        std_init=-9,
        bayesian_bias=False,
    ):
        super().__init__()
        self.n_in = in_features
        self.n_out = out_features
        self.map = use_map
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

    def kl_loss(self) -> torch.Tensor:
        r"""Compute KL divergence between posterior and prior.
        KL = \int q(w) log(q(w)/p(w)) dw
        where q(w) is the posterior and p(w) is the prior.
        """
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


class StackedLinear(nn.Module):
    "Efficient implementation of linear layers for ensembles of networks"

    def __init__(self, in_features, out_features, members, init="kaiming"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.members = members
        self.init = init
        self.weight = nn.Parameter(torch.empty((members, out_features, in_features)))
        self.bias = nn.Parameter(torch.empty((members, out_features)))
        self.reset_parameters()

    def reset_parameters(self):
        if self.init == "same":
            torch.nn.init.kaiming_uniform_(self.weight[0], a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias[0], -bound, bound)
            for i in range(self.members):
                with torch.no_grad():
                    self.weight[i].copy_(self.weight[0])
                    self.bias[i].copy_(self.bias[0])
        elif self.init == "kaiming":
            for i in range(self.members):
                torch.nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(self.bias[i], -bound, bound)
        elif self.init == "vblinear":
            for i in range(self.members):
                stdv = 1.0 / math.sqrt(self.weight.size(2))
                self.weight[i].data.normal_(0, stdv)
                self.bias[i].data.zero_()
        else:
            raise ValueError("Unknown initialization")

    def forward(self, input):
        return torch.baddbmm(self.bias[:, None, :], input, self.weight.transpose(1, 2))
