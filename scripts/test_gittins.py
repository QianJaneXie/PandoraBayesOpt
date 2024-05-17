import torch
import botorch
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition import ExpectedImprovement
from pandora_bayesopt.acquisition.gittins import GittinsIndex
from pandora_bayesopt.bayesianoptimizer import BayesianOptimizer

from botorch.models import SingleTaskGP
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from botorch.sampling.pathwise import draw_kernel_feature_paths

# use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set default tensor type to float64
torch.set_default_dtype(torch.float64)

seed = 0
torch.manual_seed(seed)

nu = 2.5
lengthscale = 0.1
outputscale = 1.0
dim = 4

base_kernel = MaternKernel(nu=nu).double()
base_kernel.lengthscale = torch.tensor([[lengthscale]])
scale_kernel = ScaleKernel(base_kernel).double()
scale_kernel.outputscale = torch.tensor([[outputscale]])

# Define Noise Level
noise_level = 1e-4

# Initialize Placeholder Data with Correct Dimensions
num_samples = 1  # Replace with actual number of samples
num_features = dim  # Replace with actual number of features
train_X = torch.zeros(num_samples, num_features)  # Placeholder data
train_Y = torch.zeros(num_samples, 1)             # Placeholder data
Yvar = torch.ones(num_samples) * noise_level

# Initialize Model
model = SingleTaskGP(train_X, train_Y, likelihood = FixedNoiseGaussianLikelihood(noise=Yvar), covar_module=scale_kernel)

# Draw a sample path
sample_path = draw_kernel_feature_paths(model, sample_shape=torch.Size([1]))
def objective_function(x):
    return sample_path(x).squeeze(0).detach()

# Set up the kernel
base_kernel = MaternKernel(nu=nu).double()
base_kernel.lengthscale = lengthscale
base_kernel.raw_lengthscale.requires_grad = False
scale_kernel = ScaleKernel(base_kernel).double()
scale_kernel.outputscale = torch.tensor([[outputscale]])
scale_kernel.raw_outputscale.requires_grad = False

dim = 4
bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
init_x = draw_sobol_samples(bounds=bounds, n=1, q=dim+1).squeeze(0)

Optimizer = BayesianOptimizer(
        objective=objective_function, 
        dim=dim, 
        maximize=True, 
        initial_points=init_x,
        input_standardize=True
    )

Optimizer.run(
    num_iterations = 2, 
    acquisition_function_class = GittinsIndex,
    lmbda = 0.0001,
    bisection_early_stopping = True
)