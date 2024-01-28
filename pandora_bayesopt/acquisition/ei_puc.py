import torch
from botorch.acquisition import AnalyticAcquisitionFunction, ExpectedImprovement
from botorch.models.model import Model

class ExpectedImprovementWithCost(AnalyticAcquisitionFunction):
    """
    This is the acquisition function EI(x) / c(x) ^ alpha, where alpha is a decay
    factor that reduces or increases the emphasis of the cost function c(x).
    """

    def __init__(self, model, best_f, maximize, cost, alpha=1.0):
        super().__init__(model=model)
        self.model = model
        self.cost = cost
        self.ei = ExpectedImprovement(model=model, best_f=best_f, maximize=maximize)
        self.alpha = alpha

    def forward(self, X):
        return self.ei(X) / torch.pow(self.cost(X).view(self.ei(X).shape), self.alpha)