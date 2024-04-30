from typing import Callable, Optional
import torch
from torch import Tensor
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.multi_step_lookahead import warmstart_multistep
from .acquisition.gittins import GittinsIndex
from .acquisition.ei_puc import ExpectedImprovementWithCost
from .acquisition.multi_step_ei import MultiStepLookaheadEI
from .acquisition.budgeted_multi_step_ei import BudgetedMultiStepLookaheadEI
from botorch.sampling.pathwise import draw_matheron_paths
from botorch.utils.sampling import optimize_posterior_samples
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.optim import optimize_acqf
from copy import copy
from .utils import fit_gp_model
import numpy as np
import math
import time

class BayesianOptimizer:
    DEFAULT_COST = torch.tensor(1.0)  # Default cost if not provided

    def __init__(self,
                 dim: int, 
                 maximize: bool, 
                 initial_points: Tensor, 
                 objective: Optional[Callable] = None, 
                 cost: Optional[Callable] = None, 
                 objective_cost: Optional[Callable] = None, 
                 input_standardize: bool = False, 
                 kernel: Optional[torch.nn.Module] = None
                ):
        self.validate_functions(objective, objective_cost)
        self.initialize_attributes(objective, cost, objective_cost, dim, maximize, initial_points, input_standardize, kernel)
    
    def validate_functions(self, objective, objective_cost):
        # Make sure that the objective function and the cost function are passed in the correct form
        if objective_cost is None and objective is None:
            raise ValueError("At least one of 'objective' or 'objective_cost' must be provided.")
        if objective is not None and objective_cost is not None:
            raise ValueError("Only one of 'objective' or 'objective_cost' can be provided.")
        self.unknown_cost = callable(objective_cost)
    
    def initialize_attributes(self, objective, cost, objective_cost, dim, maximize, initial_points, input_standardize, kernel):
        self.objective = objective
        self.cost = cost if cost is not None else self.DEFAULT_COST
        self.objective_cost = objective_cost
        self.dim = dim
        self.maximize = maximize
        self.bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
        self.best_f = None
        self.best_history = []
        self.cumulative_cost = 0.0
        self.cost_history = [0.0]
        self.runtime_history = []
        self.initialize_points(initial_points)
        self.suggested_x_full_tree = None

        # GP model parameters
        self.input_standardize = input_standardize
        self.kernel = kernel

    def initialize_points(self, initial_points):
        self.x = initial_points
        if callable(self.objective):
            self.y = self.objective(initial_points)
            if callable(self.cost):
                self.c = self.cost(initial_points)
            else:
                self.c = self.DEFAULT_COST
        if callable(self.objective_cost):
            self.y, self.c = self.objective_cost(initial_points)
        self.update_best()

    def update_best(self):
        self.best_f = self.y.max().item() if self.maximize else self.y.min().item()
        self.best_history.append(self.best_f)

    def iterate(self, acquisition_function_class, **acqf_kwargs):
        
        is_ms = False
        if acquisition_function_class == "RandomSearch":

            new_point = torch.rand(1, self.dim)
        
        else:
            if acquisition_function_class in (ExpectedImprovementWithCost, GittinsIndex, BudgetedMultiStepLookaheadEI):
                model = fit_gp_model(
                    X=self.x.detach(), 
                    objective_X=self.y.detach(), 
                    cost_X=self.c.detach(), 
                    unknown_cost=self.unknown_cost, 
                    input_standardize=self.input_standardize, 
                    kernel=self.kernel
                )
            else:
                model = fit_gp_model(
                    X=self.x.detach(), 
                    objective_X=self.y.detach(), 
                    cost_X=self.c.detach(), 
                    unknown_cost=False, 
                    input_standardize=self.input_standardize, 
                    kernel=self.kernel
                )

            acqf_args = {'model': model}
        
            if acquisition_function_class == "ThompsonSampling":
            
                # Draw sample path(s)
                paths = draw_matheron_paths(model, sample_shape=torch.Size([1]))
                
                # Optimize
                new_point, new_point_TS = optimize_posterior_samples(paths=paths, bounds=self.bounds, maximize=self.maximize)

                self.current_acq = new_point_TS.item()

            else:
                
                if acquisition_function_class == qPredictiveEntropySearch:

                    # Draw sample path(s)
                    paths = draw_matheron_paths(model, sample_shape=torch.Size([1]))
                    
                    # Optimize
                    optimal_input, _ = optimize_posterior_samples(paths=paths, bounds=self.bounds, maximize=self.maximize)

                    acqf_args['optimal_inputs'] = optimal_input
                    acqf_args['maximize'] = self.maximize

                elif acquisition_function_class == UpperConfidenceBound:
                    if acqf_kwargs.get('heuristic') == True:
                        acqf_args['beta'] = 2*np.log(self.dim*((self.cumulative_cost+1)**2)*(math.pi**2)/(6*0.1))/5
                    else:
                        acqf_args['beta'] = acqf_kwargs['beta']
                    acqf_args['maximize'] = self.maximize
                
                elif acquisition_function_class == ExpectedImprovement:
                    acqf_args['best_f'] = self.best_f
                    acqf_args['maximize'] = self.maximize

                elif acquisition_function_class == ExpectedImprovementWithCost:
                    acqf_args['best_f'] = self.best_f
                    acqf_args['maximize'] = self.maximize
                    acqf_args['cost'] = self.cost
                    acqf_args['unknown_cost'] = self.unknown_cost
                    if acqf_kwargs.get('cost_cooling') == True:
                        cost_exponent = (self.budget - self.cumulative_cost) / self.budget
                        cost_exponent = max(cost_exponent, 0)  # Ensure cost_exponent is non-negative
                        acqf_args['cost_exponent'] = cost_exponent

                elif acquisition_function_class == GittinsIndex:
                    acqf_args['maximize'] = self.maximize
                    if acqf_kwargs.get('step_EIpu') == True:
                        if self.need_lmbda_update:
                            if callable(self.cost) or callable(self.objective_cost):
                                # Optimize EIpu first to get new_point_EIpu
                                EIpu = ExpectedImprovementWithCost(model=model, best_f=self.best_f, maximize=self.maximize, cost=self.cost, unknown_cost=self.unknown_cost)
                                _, new_point_EIpu = optimize_acqf(
                                    acq_function=EIpu,
                                    bounds=self.bounds,
                                    q=1,
                                    num_restarts=10*self.dim,
                                    raw_samples=200*self.dim,
                                    options={'method': 'L-BFGS-B'},
                                )
                                if self.current_lmbda == None:
                                    self.current_lmbda = new_point_EIpu.item() / 2
                                else:
                                    self.current_lmbda = min(self.current_lmbda, new_point_EIpu.item() / 2)

                            else:
                                # Optimize EI first to get new_point_EI
                                EI = ExpectedImprovement(model=model, best_f=self.best_f, maximize=self.maximize)
                                _, new_point_EI = optimize_acqf(
                                    acq_function=EI,
                                    bounds=self.bounds,
                                    q=1,
                                    num_restarts=10*self.dim,
                                    raw_samples=200*self.dim,
                                    options={'method': 'L-BFGS-B'},
                                )
                                if self.current_lmbda == None:
                                    self.current_lmbda = new_point_EI.item() / 2
                                else:
                                    self.current_lmbda = min(self.current_lmbda, new_point_EI.item() / 2)
                            self.need_lmbda_update = False  # Reset the flag
                        print("current lambda:", self.current_lmbda)
                        acqf_args['lmbda'] = self.current_lmbda
                        self.lmbda_history.append(self.current_lmbda)

                    elif acqf_kwargs.get('step_divide') == True:
                        if self.need_lmbda_update:
                            self.current_lmbda = self.current_lmbda / acqf_kwargs.get('alpha')
                            self.need_lmbda_update = False
                        acqf_args['lmbda'] = self.current_lmbda
                        self.lmbda_history.append(self.current_lmbda)

                    else: 
                        acqf_args['lmbda'] = acqf_kwargs['lmbda']

                    acqf_args['cost'] = self.cost
                    acqf_args['unknown_cost'] = self.unknown_cost
                elif acquisition_function_class == MultiStepLookaheadEI:
                    is_ms = True
                    acqf_args['batch_size'] = 1
                    acqf_args['lookahead_batch_sizes'] = [1, 1, 1]
                    acqf_args['num_fantasies'] = [1, 1, 1]
                elif acquisition_function_class == BudgetedMultiStepLookaheadEI:
                    is_ms = True
                    acqf_args['cost_function'] = copy(self.cost)
                    acqf_args['unknown_cost'] = self.unknown_cost
                    acqf_args['budget_plus_cumulative_cost'] = min(self.budget - self.cumulative_cost, self.c[-4:].sum().item()) + self.c.sum().item()
                    print(acqf_args['budget_plus_cumulative_cost'])
                    acqf_args['batch_size'] = 1
                    acqf_args['lookahead_batch_sizes'] = [1, 1, 1]
                    acqf_args['num_fantasies'] = [1, 1, 1]
                else:
                    acqf_args.update(**acqf_kwargs)
                    
                acq_function = acquisition_function_class(**acqf_args)
                if self.suggested_x_full_tree is not None:
                    batch_initial_conditions = warmstart_multistep(
                            acq_function=acq_function,
                            bounds=self.bounds,
                            num_restarts=10 * self.dim,
                            raw_samples=200 * self.dim,
                            full_optimizer=self.suggested_x_full_tree,
                            algo_params=acqf_args,
                        )
                else:
                    batch_initial_conditions = None
                q = acq_function.get_augmented_q_batch_size(1) if is_ms else 1
                candidates, candidates_acq_vals = optimize_acqf(
                    acq_function=acq_function,
                    bounds=self.bounds,
                    q=q,
                    num_restarts=10 * self.dim,
                    raw_samples=200 * self.dim,
                    options={
                            "batch_limit": 5,
                            "maxiter": 200,
                            "method": "L-BFGS-B",
                        },
                    batch_initial_conditions=batch_initial_conditions,
                    return_best_only=False,
                    return_full_tree=is_ms,
                )

                candidates =  candidates.detach()
                
                if is_ms:
                    # save all tree variables for multi-step initialization
                    self.suggested_x_full_tree = candidates.clone()
                    candidates = acq_function.extract_candidates(candidates)

                best_idx = torch.argmax(candidates_acq_vals.view(-1), dim=0)
                new_point = candidates[best_idx]
                self.current_acq = candidates_acq_vals[best_idx].item()
        
        if self.unknown_cost:
            new_value, new_cost = self.objective_cost(new_point.detach())
        else: 
            new_value = self.objective(new_point.detach())
        self.x = torch.cat((self.x, new_point.detach()))
        self.y = torch.cat((self.y, new_value.detach()))
        self.update_best()
        self.update_cost(new_point)

        if acquisition_function_class == "RandomSearch":
            self.current_acq = new_value.item()

        # Check if lmbda needs to be updated in the next iteration
        if acquisition_function_class == GittinsIndex and (acqf_kwargs.get('step_EIpu') == True or acqf_kwargs.get('step_divide') == True):
            if (self.maximize and self.current_acq < self.best_f) or (not self.maximize and -self.current_acq > self.best_f):
                self.need_lmbda_update = True


    def update_cost(self, new_point):
        if callable(self.cost):
            # If self.cost is a function, call it and update cumulative cost
            new_cost = self.cost(new_point)
            self.c = torch.cat((self.c, new_cost))
            self.cumulative_cost += new_cost.sum().item()
        elif callable(self.objective_cost):
            new_value, new_cost = self.objective_cost(new_point)
            self.c = torch.cat((self.c, new_cost))
            self.cumulative_cost += new_cost.sum().item()
        else:
            # If self.cost is not a function, just increment cumulative cost by self.cost
            self.cumulative_cost += self.cost.item()

        self.cost_history.append(self.cumulative_cost)

    def print_iteration_info(self, iteration):
        print(f"Iteration {iteration}, New point: {self.x[-1].squeeze().detach().numpy()}, New value: {self.y[-1].detach().numpy()}")
        print("Best observed value:", self.best_f)
        print("Current acquisition value:", self.current_acq)
        print("Cumulative cost:", self.cumulative_cost)
        if hasattr(self, 'need_lmbda_update'):
            print("Gittins lmbda:", self.lmbda_history[-1])
        print("Running time:", self.runtime)
        print()

    def run(self, num_iterations, acquisition_function_class, **acqf_kwargs):
        self.budget = num_iterations
        if acquisition_function_class == GittinsIndex:
            if acqf_kwargs.get('step_EIpu') == True:
                self.current_lmbda = None
                self.need_lmbda_update = True
                self.lmbda_history = []
            if acqf_kwargs.get('step_divide') == True:
                self.current_lmbda = 0.1
                self.need_lmbda_update = False
                self.lmbda_history = []                

        for i in range(num_iterations):
            start = time.process_time()
            self.iterate(acquisition_function_class, **acqf_kwargs)
            end = time.process_time()
            runtime = end - start
            self.runtime = runtime
            self.runtime_history.append(runtime)
            self.print_iteration_info(i)

    def run_until_budget(self, budget, acquisition_function_class, **acqf_kwargs):
        self.budget = budget
        if acquisition_function_class == GittinsIndex:
            if acqf_kwargs.get('step_EIpu') == True:
                self.current_lmbda = None
                self.need_lmbda_update = True
                self.lmbda_history = []
            if acqf_kwargs.get('step_divide') == True:
                self.current_lmbda = 0.1
                self.need_lmbda_update = False
                self.lmbda_history = []  

        i = 0
        while self.cumulative_cost < self.budget:
            start = time.process_time()
            self.iterate(acquisition_function_class, **acqf_kwargs)
            end = time.process_time()
            runtime = end - start
            self.runtime = runtime
            self.runtime_history.append(runtime)
            self.print_iteration_info(i)
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

    def get_lmbda_history(self):
        return self.lmbda_history
    
    def get_runtime_history(self):
        return self.runtime_history