#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import time
import torch
import warnings
import toml

from abc import ABC, abstractmethod

from botorch.acquisition import AnalyticAcquisitionFunction, ExpectedImprovement
from src.acquisition import GittinsIndex
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Log
from botorch.optim import optimize_acqf
from botorch.utils import standardize
from botorch.utils.sampling import draw_sobol_samples

from gpytorch.mlls import ExactMarginalLogLikelihood

# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

warnings.filterwarnings("ignore")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
tkwargs = {
    "device": device,
    "dtype": torch.double,
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")
seed = int(os.getenv('MY_SEED', '111'))  # Default to 111 if MY_SEED is not set
print("SMOKE_TEST:", SMOKE_TEST)
print("seed:", seed)
print()

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the results directory relative to the script directory
results_dir = os.path.join(script_dir, 'results/NoisyLinearLeastSquares')

# Create the results directory if it doesn't exist
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

result_filename = f'trial_{seed}.toml'
result_filepath = os.path.join(results_dir, result_filename)

def save_results_to_file(data, filepath):
    with open(filepath, 'w') as file:
        toml.dump(data, file)  # or use 'json.dump(data, file, indent=4)'


# In[3]:


class ExpectedImprovementWithCost(AnalyticAcquisitionFunction):
    """
    This is the acquisition function EI(x) / c(x) ^ alpha, where alpha is a decay
    factor that reduces or increases the emphasis of the cost model c(x).
    """

    def __init__(self, model, best_f, cost_model, alpha=1):
        super().__init__(model=model)
        self.model = model
        self.cost_model = cost_model
        self.ei = ExpectedImprovement(model=model, best_f=best_f)
        self.alpha = alpha

    def forward(self, X):
        return self.ei(X) / torch.pow(self.cost_model(X)[:, 0], self.alpha)


# In[4]:


class NoisyLinearLeastSquares:
    """
    The standard linear least squares problem min_x ||Ax - b||_2.
    We compute the loss via batching that introduces noise.
    """

    def __init__(self, A, b, batch_size=50):
        self.A = A
        self.b = b
        self.batch_size = min(batch_size, self.A.shape[0])

    def fit(self, lr=1, niters=100):
        x = torch.zeros(A.shape[1], 1, requires_grad=True, **tkwargs)
        optimizer = torch.optim.Adam([x], lr=lr)
        batch_indices = torch.randperm(A.shape[1])[: self.batch_size]
        for i in range(niters):
            res = torch.matmul(self.A[batch_indices, :], x) - self.b[batch_indices]
            loss = torch.norm(res)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return x, loss


# In[5]:


n = 30000 if not SMOKE_TEST else 300
d = 3000 if not SMOKE_TEST else 30
A = torch.rand(n, d, **tkwargs)
b = torch.rand(n, 1, **tkwargs)


# In[7]:


# Assume x0 is learning rate, x1 is batch_size, x2 is iterations
bounds = torch.tensor([[0.05, 40, 10], [1, 1000, 400]], **tkwargs)


def objective(x):
    learning_rate = x[0]
    batch_size = int(x[1])
    num_iters = int(x[2])
    model = NoisyLinearLeastSquares(A, b, batch_size=batch_size)
    t_start = time.time()
    x, loss = model.fit(lr=learning_rate, niters=num_iters)
    cost = time.time() - t_start
    return loss.item(), cost


class CostModel(torch.nn.Module, ABC):
    """
    Simple abstract class for a cost model.
    """    
    
    @abstractmethod
    def forward(self, X):
        pass
    
# Simplified cost model based on analysis above
class LinearCostModel(CostModel):
    def __init__(self):
        super().__init__()

    # Assume x1 is batch_size, x2 is iterations
    def forward(self, X):
        return X[:, :, 1] * X[:, :, 2]


def generate_initial_data(obj, bounds, num, seed):
    dim = bounds.shape[1]
    train_x = draw_sobol_samples(bounds=bounds, n=num, q=1, seed=seed).squeeze(1)
    train_y = []
    cost_y = []
    for x in train_x:
        y, c = obj(x)
        train_y.append(y)
        cost_y.append(c)
    return (
        train_x,
        torch.tensor(train_y, **tkwargs).unsqueeze(-1),
        torch.tensor(cost_y, **tkwargs).unsqueeze(-1),
    )


# Generate initial data
num_iterations = 25
num_initial = 5
init_X, init_Y, init_C = generate_initial_data(objective, bounds, num_initial, seed)


# In[10]:

print("EIpu")
train_X = init_X
train_Y = init_Y
cost_Y = init_C

for i in range(num_iterations):

    # Train GP
    train_Y_flip = -1 * standardize(train_Y)  # we want to minimize so we negate
    gp = SingleTaskGP(train_X=train_X, train_Y=train_Y_flip)
    mll = ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp)
    fit_gpytorch_mll(mll)

    # Train Cost Model
    cost_model = LinearCostModel()
    fmax = torch.max(train_Y_flip)
    eipu = ExpectedImprovementWithCost(
        model=gp,
        best_f=fmax,
        cost_model=cost_model,
        alpha=1.0,
    )
    new_x, acq_value = optimize_acqf(
        acq_function=eipu,
        bounds=bounds,
        q=1,
        num_restarts=20,
        raw_samples=1024,
    )

    # Get objective value and cost
    new_y, cost_y = objective(new_x.squeeze())

    # update training points
    train_X = torch.cat([train_X, new_x])
    train_Y = torch.cat([train_Y, torch.tensor([new_y], **tkwargs).unsqueeze(1)])
    cost_Y = torch.cat([cost_Y, torch.tensor([cost_y], **tkwargs).unsqueeze(1)])
        
    
costs_eipu = cost_Y[:, 0]
results_eipu, _ = torch.cummin(train_Y, dim=0)
times_eipu = torch.cumsum(costs_eipu, dim=0)

print("cumulative time:", times_eipu)
print("loss:", results_eipu)
print()


print("EIpu with cost-cooling")
train_X = init_X
train_Y = init_Y
cost_Y = init_C

for i in range(num_iterations):
    alpha = (num_iterations - i - 1) / (num_iterations - 1)

    # Train GP
    train_Y_flip = -1 * standardize(train_Y)  # we want to minimize so we negate
    gp = SingleTaskGP(train_X=train_X, train_Y=train_Y_flip)
    mll = ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp)
    fit_gpytorch_mll(mll)

    # Train Cost Model
    cost_model = LinearCostModel()
    fmax = torch.max(train_Y_flip)
    eipuc = ExpectedImprovementWithCost(
        model=gp,
        best_f=fmax,
        cost_model=cost_model,
        alpha=alpha,
    )
    new_x, acq_value = optimize_acqf(
        acq_function=eipuc,
        bounds=bounds,
        q=1,
        num_restarts=20,
        raw_samples=1024,
    )

    # Get objective value and cost
    new_y, cost_y = objective(new_x.squeeze())

    # update training points
    train_X = torch.cat([train_X, new_x])
    train_Y = torch.cat([train_Y, torch.tensor([new_y], **tkwargs).unsqueeze(1)])
    cost_Y = torch.cat([cost_Y, torch.tensor([cost_y], **tkwargs).unsqueeze(1)])
        
    
costs_eipuc = cost_Y[:, 0]
results_eipuc, _ = torch.cummin(train_Y, dim=0)
times_eipuc = torch.cumsum(costs_eipuc, dim=0)

print("cumulative time:", times_eipuc)
print("loss:", results_eipuc)
print()


# In[11]:

print("EI")
train_X = init_X
train_Y = init_Y
cost_Y = init_C

for i in range(num_iterations):
    # Train GP
    train_Y_flip = -1 * standardize(train_Y)  # we want to minimize so we negate
    gp = SingleTaskGP(train_X=train_X, train_Y=train_Y_flip)
    mll = ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp)
    fit_gpytorch_mll(mll)

    # Train Cost Model
    fmax = torch.max(train_Y_flip)
    ei = ExpectedImprovement(gp, fmax)
    new_x, acq_value = optimize_acqf(
        acq_function=ei,
        bounds=bounds,
        q=1,
        num_restarts=20,
        raw_samples=1024,
    )

    # Get objective value and cost
    new_y, cost_y = objective(new_x.squeeze())

    # update training points
    train_X = torch.cat([train_X, new_x])
    train_Y = torch.cat([train_Y, torch.tensor([new_y], **tkwargs).unsqueeze(1)])
    cost_Y = torch.cat([cost_Y, torch.tensor([cost_y], **tkwargs).unsqueeze(1)])

costs_ei = cost_Y[:, 0]
results_ei, _ = torch.cummin(train_Y, dim=0)
times_ei = torch.cumsum(costs_ei, dim=0)

print("cumulative time:", times_ei)
print("loss:", results_ei)
print()


# In[ ]:

print("Gittins")
train_X = init_X
train_Y = init_Y
cost_Y = init_C

for i in range(num_iterations):
    # Train GP
    train_Y_flip = -1 * standardize(train_Y)  # we want to minimize so we negate
    gp = SingleTaskGP(train_X=train_X, train_Y=train_Y_flip)
    mll = ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp)
    fit_gpytorch_mll(mll)

    # Train Cost Model
    cost_model = LinearCostModel()
    gi = GittinsIndex(model=gp, lmbda=0.0001, cost=cost_model)
    new_x, acq_value = optimize_acqf(
        acq_function=gi,
        bounds=bounds,
        q=1,
        num_restarts=20,
        raw_samples=1024,
    )

    # Get objective value and cost
    new_y, cost_y = objective(new_x.squeeze())

    # update training points
    train_X = torch.cat([train_X, new_x])
    train_Y = torch.cat([train_Y, torch.tensor([new_y], **tkwargs).unsqueeze(1)])
    cost_Y = torch.cat([cost_Y, torch.tensor([cost_y], **tkwargs).unsqueeze(1)])
        

costs_gi = cost_Y[:, 0]
results_gi, _ = torch.cummin(train_Y, dim=0)
times_gi = torch.cumsum(costs_gi, dim=0)

print("cumulative time:", times_gi)
print("loss:", results_gi)
print()


# In[ ]:

print("Gittins with lambda schedule")
train_X = init_X
train_Y = init_Y
cost_Y = init_C

for i in range(num_iterations):
    # Train GP
    train_Y_flip = -1 * standardize(train_Y)  # we want to minimize so we negate
    gp = SingleTaskGP(train_X=train_X, train_Y=train_Y_flip)
    mll = ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp)
    fit_gpytorch_mll(mll)

    # Train Cost Model
    cost_model = LinearCostModel()
    fmax = torch.max(train_Y_flip)
    eipu = ExpectedImprovementWithCost(
        model=gp,
        best_f=fmax,
        cost_model=cost_model,
        alpha=1.0,
    )
    _, eipu_max = optimize_acqf(
        acq_function=eipu,
        bounds=bounds,
        q=1,
        num_restarts=20,
        raw_samples=1024,
    )
    gi_schedule = GittinsIndex(model=gp, lmbda=eipu_max/2, cost=cost_model)
    new_x, acq_value = optimize_acqf(
        acq_function=gi_schedule,
        bounds=bounds,
        q=1,
        num_restarts=20,
        raw_samples=1024,
    )

    # Get objective value and cost
    new_y, cost_y = objective(new_x.squeeze())

    # update training points
    train_X = torch.cat([train_X, new_x])
    train_Y = torch.cat([train_Y, torch.tensor([new_y], **tkwargs).unsqueeze(1)])
    cost_Y = torch.cat([cost_Y, torch.tensor([cost_y], **tkwargs).unsqueeze(1)])

costs_gi_schedule = cost_Y[:, 0]
results_gi_schedule, _ = torch.cummin(train_Y, dim=0)
times_gi_schedule = torch.cumsum(costs_gi_schedule, dim=0)

print("cumulative time:", times_gi_schedule)
print("loss:", results_gi_schedule)

# Data to be saved
result_data = {
    'problem': {
        'objective': 'NoisyLinearLeastSquares',
        'cost': 'LinearCostModel',
    },
    'trial': seed,
    'number of iterations': num_iterations,
    'cost history': {
        'EI': costs_ei.numpy(),
        'EIpu': costs_eipu.numpy(),
        'EIpu (cost-cooling)': costs_eipuc.numpy(),
        'Gittins (lmbda=0.0001)': costs_gi.numpy(),
        'Gittins (lmbda=EIpu_max/2)': costs_gi_schedule.numpy()
    },
    'cumulative time history': {
        'EI': times_ei.numpy(),
        'EIpu': times_eipu.numpy(),
        'EIpu (cost-cooling)': times_eipuc.numpy(),
        'Gittins (lmbda=0.0001)': times_gi.numpy(),
        'Gittins (lmbda=EIpu_max/2)': times_gi_schedule.numpy()
    },
    'loss history': {
        'EI': results_ei.numpy(),
        'EIpu': results_eipu.numpy(),
        'EIpu (cost-cooling)': results_eipuc.numpy(),
        'Gittins (lmbda=0.0001)': results_gi.numpy(),
        'Gittins (lmbda=EIpu_max/2)': results_gi_schedule.numpy()
    }
}

# Save to file
save_results_to_file(result_data, result_filepath)


