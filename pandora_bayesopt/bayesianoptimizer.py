#!/usr/bin/env python3

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
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy,qMultiFidelityMaxValueEntropy
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.models.deterministic import GenericDeterministicModel
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
                 kernel: Optional[torch.nn.Module] = None,
                 noisy_observation: bool = False,
                 noise_level: Optional[float] = 0.1,
                 output_standardize: bool = False,
                ):
        self.validate_functions(objective, objective_cost)
        self.initialize_attributes(objective, cost, objective_cost, dim, maximize, initial_points, kernel, noisy_observation, noise_level, output_standardize)


    def validate_functions(self, objective, objective_cost):
        # Make sure that the objective function and the cost function are passed in the correct form
        if objective_cost is None and objective is None:
            raise ValueError("At least one of 'objective' or 'objective_cost' must be provided.")
        if objective is not None and objective_cost is not None:
            raise ValueError("Only one of 'objective' or 'objective_cost' can be provided.")
        self.unknown_cost = callable(objective_cost)


    def initialize_attributes(self, objective, cost, objective_cost, dim, maximize, initial_points, kernel, noisy_observation, noise_level, output_standardize):
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
        self.acq_history = [np.nan]
        self.runtime_history = []

        # GP model parameters
        self.kernel = kernel
        self.noisy_observation = noisy_observation
        self.noise_level = noise_level
        self.output_standardize = output_standardize
        
        self.suggested_x_full_tree = None
        self.initialize_points(initial_points)

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
        if self.noisy_observation:
            noise = torch.randn_like(self.y) * self.noise_level
            self.y += noise
        self.acq_history.append(self.best_f)  # make sure the length of acq_history is the same as best_history
        self.update_best()


    def update_best(self):
        self.best_f = self.y.max().item() if self.maximize else self.y.min().item()
        self.best_history.append(self.best_f)


    def iterate(self, acquisition_function_class, **acqf_kwargs):
        
        is_rs = False
        is_ms = False
        is_ts = False
        is_pes = False

        if acquisition_function_class == BudgetedMultiStepLookaheadEI:
            gaussian_likelihood = True
        else:
            gaussian_likelihood = False
        
        if acquisition_function_class == "RandomSearch":
            is_rs = True
            new_point = torch.rand(1, self.dim)
            
        else:
            if acquisition_function_class in (ExpectedImprovementWithCost, GittinsIndex, BudgetedMultiStepLookaheadEI):
                unknown_cost = self.unknown_cost
            else:
                unknown_cost = False

            model = fit_gp_model(
                X=self.x.detach(), 
                objective_X=self.y.detach(), 
                cost_X=self.c.detach(), 
                unknown_cost=unknown_cost,
                kernel=self.kernel,
                gaussian_likelihood=gaussian_likelihood,
                noisy_observation=self.noisy_observation,
                output_standardize=self.output_standardize,
            )

            acqf_args = {'model': model}
        
            
            if acquisition_function_class in ("ThompsonSampling", qPredictiveEntropySearch, "SurrogatePrice"):
            
                # Draw sample path(s)
                paths = draw_matheron_paths(model, sample_shape=torch.Size([1]))
                
                # Optimize
                optimal_input, optimal_output = optimize_posterior_samples(paths=paths, bounds=self.bounds, maximize=self.maximize)

                if acquisition_function_class == "ThompsonSampling":
                    is_ts = True
                    new_point = optimal_input
                    self.current_acq = optimal_output.item()
                
                elif acquisition_function_class == qPredictiveEntropySearch:
                    is_pes = True
                    PES = qPredictiveEntropySearch(model=model, optimal_inputs=optimal_input, maximize=self.maximize)
                    new_point, new_point_PES = optimize_acqf(
                        acq_function=PES,
                        bounds=self.bounds,
                        q=1,
                        num_restarts=10*self.dim,
                        raw_samples=200*self.dim,
                        options={
                                "batch_limit": 5,
                                "maxiter": 200,
                                "with_grad": False
                            },
                    )
                    self.current_acq = new_point_PES.item()

            
            if acquisition_function_class == GittinsIndex:
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


            elif acquisition_function_class in (qMaxValueEntropy, qMultiFidelityMaxValueEntropy):
                candidate_set = torch.rand(1000*self.dim, self.bounds.size(1))
                candidate_set = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * candidate_set
                acqf_args['candidate_set'] = candidate_set

                if acquisition_function_class == qMultiFidelityMaxValueEntropy:
                    cost_function = copy(self.cost)
                    class CostModel(GenericDeterministicModel):
                        def __init__(self):
                            super().__init__(f=cost_function)
                    cost_model = CostModel()
                    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
                    acqf_args['cost_aware_utility'] = cost_aware_utility


            elif acquisition_function_class in (MultiStepLookaheadEI, BudgetedMultiStepLookaheadEI):
                is_ms = True
                acqf_args['batch_size'] = 1
                acqf_args['lookahead_batch_sizes'] = [1, 1, 1]
                acqf_args['num_fantasies'] = [1, 1, 1]
                
                if acquisition_function_class == BudgetedMultiStepLookaheadEI:
                    acqf_args['cost_function'] = copy(self.cost)
                    acqf_args['unknown_cost'] = self.unknown_cost
                    acqf_args['budget_plus_cumulative_cost'] = min(self.budget - self.cumulative_cost, self.c[-4:].sum().item()) + self.c.sum().item()
                    print(acqf_args['budget_plus_cumulative_cost'])
                
            
            else:
                acqf_args.update(**acqf_kwargs)


            if is_ts == False and is_pes == False:
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
                best_point = candidates[best_idx]
                best_acq_val = candidates_acq_vals[best_idx].item()

                new_point = best_point
                self.current_acq = best_acq_val


        if self.unknown_cost:
            new_value, new_cost = self.objective_cost(new_point.detach())
        else: 
            new_value = self.objective(new_point.detach())

        if self.noisy_observation:
            noise = torch.randn_like(new_value) * self.noise_level
            new_value += noise

        self.x = torch.cat((self.x, new_point.detach()))
        self.y = torch.cat((self.y, new_value.detach()))
        self.update_best()
        self.update_cost(new_point)

        if is_rs:
            self.current_acq = new_value.item()

        self.acq_history.append(self.current_acq)

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
                self.current_lmbda = acqf_kwargs['init_lmbda']
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
                self.current_lmbda = acqf_kwargs['init_lmbda']
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


    def get_acq_history(self):
        return self.acq_history


    def get_runtime_history(self):
        return self.runtime_history