#!/usr/bin/env python3

import torch
import gpytorch

# Set default tensor type to float64
torch.set_default_dtype(torch.float64)

class VariableAmplitudeKernel(gpytorch.kernels.Kernel):
    def __init__(self, base_kernel, amplitude_function):
        super(VariableAmplitudeKernel, self).__init__()
        self.base_kernel = base_kernel
        self.amplitude_function = amplitude_function

    def forward(self, x1, x2=None, diag=False, last_dim_is_batch=False, **params):
        
        base_covar = self.base_kernel(x1, x2, diag=diag, **params)
        amplitude_x1 = self.amplitude_function(x1)
        amplitude_x2 = self.amplitude_function(x2)
        
        assert not last_dim_is_batch

        # Convert to tensors if necessary
        if not isinstance(amplitude_x1, torch.Tensor):
            amplitude_x1 = torch.tensor(amplitude_x1)
        if not isinstance(amplitude_x2, torch.Tensor):
            amplitude_x2 = torch.tensor(amplitude_x2)

                
        if diag:
            # Ensure amplitude is 1D for diagonal case
            return base_covar * amplitude_x1
        else:
            # Reshape for non-diagonal case
            amplitude_reshaped_x1 = amplitude_x1.unsqueeze(-1)
            amplitude_reshaped_x2 = amplitude_x2.unsqueeze(-2)
            amplitude = (amplitude_reshaped_x1 * amplitude_reshaped_x2).sqrt()
            
            # Ensure the shapes are compatible for multiplication
            return base_covar * amplitude