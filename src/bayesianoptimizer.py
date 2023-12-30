import torch
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition import ExpectedImprovement
from src.acquisition import GittinsIndex, ExpectedImprovementWithCost
from botorch.optim import optimize_acqf
from src.utils import fit_gp_model

class BayesianOptimizer:
    def __init__(self, objective, dim, maximize=True, seed=None, num_points=None, cost=None, nu=2.5, lengthscale=1.0, outputscale=1.0):
        self.objective = objective
        self.maximize = maximize
        self.dim = dim
        self.num_points = 2 * dim + 1 if num_points is None else num_points
        self.bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
        self.best_f = None
        self.best_history = []
        self.cost = cost if cost is not None else 1.0
        self.cumulative_cost = 0.0
        self.cost_history = [0.0]
        torch.set_default_dtype(torch.float64)
        if seed is not None:
            torch.manual_seed(seed)
        self.initialize_points(seed)
        
        # GP model parameters
        self.nu = nu
        self.lengthscale = lengthscale
        self.outputscale = outputscale

    def initialize_points(self, seed):
        self.x = draw_sobol_samples(bounds=self.bounds, n=1, q=self.num_points, seed=seed).squeeze(0).requires_grad_(True)
        self.y = self.objective(self.x)
        self.update_best()

    def update_best(self):
        self.best_f = self.y.max().item() if self.maximize else self.y.min().item()
        self.best_history.append(self.best_f)

    def iterate(self, acquisition_function_class, lmbda=None, **acqf_kwargs):
        model = fit_gp_model(self.x.detach(), self.y.detach(), nu=self.nu, lengthscale=self.lengthscale, outputscale=self.outputscale)
        acqf_args = {'model': model, 'maximize': self.maximize}
        
        if acquisition_function_class == ExpectedImprovement:
            acqf_args['best_f'] = self.best_f

        elif acquisition_function_class == ExpectedImprovementWithCost:
            acqf_args['best_f'] = self.best_f
            acqf_args['cost'] = self.cost

        elif acquisition_function_class == GittinsIndex:
            if lmbda is None:
                if callable(self.cost):
                    # Optimize EIpu first to get new_point_EIpu
                    EIpu = ExpectedImprovementWithCost(model=model, best_f=self.best_f, maximize=self.maximize, cost=self.cost)
                    _, new_point_EIpu = optimize_acqf(
                        acq_function=EIpu,
                        bounds=self.bounds,
                        q=1,
                        num_restarts=20,
                        raw_samples=1024,
                        options={'method': 'L-BFGS-B'},
                    )
                    lmbda = new_point_EIpu.item() / 2
                else:
                    # Optimize EI first to get new_point_EI
                    EI = ExpectedImprovement(model=model, best_f=self.best_f, maximize=self.maximize)
                    _, new_point_EI = optimize_acqf(
                        acq_function=EI,
                        bounds=self.bounds,
                        q=1,
                        num_restarts=20,
                        raw_samples=1024,
                        options={'method': 'L-BFGS-B'},
                    )
                    lmbda = new_point_EI.item() / 2

            acqf_args['lmbda'] = lmbda
            acqf_args['cost'] = self.cost

        else:
            acqf_args.update(**acqf_kwargs)
            
        acq_function = acquisition_function_class(**acqf_args)

        new_point, _ = optimize_acqf(
            acq_function=acq_function,
            bounds=self.bounds,
            q=1,
            num_restarts=10*self.dim,
            raw_samples=1024*self.dim,
            options={'method': 'L-BFGS-B'},
        )
        new_value = self.objective(new_point)
        self.x = torch.cat((self.x, new_point))
        self.y = torch.cat((self.y, new_value))
        self.update_best()
        self.update_cost(new_point)

    def update_cost(self, new_point):
        if callable(self.cost):
            # If self.cost is a function, call it and update cumulative cost
            cost = self.cost(new_point)
            self.cumulative_cost += cost.sum().item()
        else:
            # If self.cost is not a function, just increment cumulative cost by self.cost
            self.cumulative_cost += self.cost

        self.cost_history.append(self.cumulative_cost)

    def print_iteration_info(self, iteration):
        print(f"Iteration {iteration}, New point: {self.x[-1].squeeze().detach().numpy()}, New value: {self.y[-1].item()}")
        print("Best observed value:", self.best_f)
        print("Cumulative cost:", self.cumulative_cost)
        print()

    def run(self, num_iterations, acquisition_function_class, lmbda=None, **acqf_kwargs):
        for i in range(num_iterations):
            self.iterate(acquisition_function_class, lmbda=lmbda, **acqf_kwargs)
            # self.print_iteration_info(i)

    def run_until_budget(self, budget, acquisition_function_class, lmbda=None, **acqf_kwargs):
        i = 0
        while self.cumulative_cost < budget:
            self.iterate(acquisition_function_class, lmbda=lmbda, **acqf_kwargs)
            # self.print_iteration_info(i)
            if self.cumulative_cost >= budget:
                break
            i += 1

    def get_best_value(self):
        return self.best_f

    def get_best_history(self):
        return self.best_history

    def get_cumulative_cost(self):
        return self.cumulative_cost

    def get_cost_history(self):
        return self.cost_history

    def get_regret_history(self, global_optimum):
        """
        Compute the regret history.

        Parameters:
        - global_optimum (float): The global optimum value of the objective function.

        Returns:
        - list: The regret history.
        """
        return [global_optimum - f if self.maximize else f - global_optimum for f in self.best_history]