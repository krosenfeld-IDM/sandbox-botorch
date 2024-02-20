# https://github.com/pytorch/botorch/blob/main/README.md

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize

train_X = torch.rand(10, 2, dtype=torch.float64)
# explicit output dimension -- Y is 10 x 1
train_Y = 1 - (train_X - 0.5).norm(dim=-1, keepdim=True)
train_Y += 0.1 * torch.rand_like(train_Y)

gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

from botorch.acquisition import UpperConfidenceBound

UCB = UpperConfidenceBound(gp, beta=0.1)

from botorch.optim import optimize_acqf

bounds = torch.stack([torch.zeros(2), torch.ones(2)])
candidate, acq_value = optimize_acqf(
    UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
)

print("done")