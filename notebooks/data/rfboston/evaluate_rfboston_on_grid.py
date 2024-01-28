import numpy as np
import os
import sys
import torch

from sklearn_problem_gen import sklearn_classifier_objective

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(script_dir[:-12])

n = 10
X = []

for a in np.linspace(0, 1, n):
    for b in np.linspace(0, 1, n):
        for c in np.linspace(0, 1, n):
            X.append([a,b,c])

X = torch.tensor(X)
X_unscaled = X.clone()
X_unscaled[:, 0] = 255.0 * X_unscaled[:, 0] + 1.0
X_unscaled[:, 1] = 63.0 * X_unscaled[:, 1] + 1.0
X_unscaled[:, 2] = X_unscaled[:, 2] - 1.0

dataset_name = "Boston"
model_name = "RandomForestReg"

objective_X, cost_X = sklearn_classifier_objective(X=X_unscaled, dataset_name=dataset_name, model_name=model_name, random_state=165)

np.savetxt(script_dir + "/rfboston_on_grid_X.txt", X.numpy())
np.savetxt(script_dir + "/rfboston_on_grid_objective.txt", objective_X.numpy())
np.savetxt(script_dir + "/rfboston_on_grid_cost.txt", cost_X.numpy())
