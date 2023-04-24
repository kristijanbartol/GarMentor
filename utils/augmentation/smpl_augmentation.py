import numpy as np
import torch


def uniform_sample_shape(batch_size, mean_shape, delta_betas_range):
    """
    Uniform sampling of shape parameter deviations from the mean.
    """
    l, h = delta_betas_range
    delta_betas = (h-l)*torch.rand(batch_size, mean_shape.shape[0], device= mean_shape.device) + l
    shape = delta_betas + mean_shape
    return shape  # (bs, num_smpl_betas)


def normal_sample_params(batch_size, mean_params, std_vector):
    """
    Gaussian sampling of shape parameter deviations from the mean.
    """
    shape = mean_params + torch.randn(batch_size, mean_params.shape[0], device=mean_params.device)*std_vector
    return shape  # (bs, num_smpl_betas)


def normal_sample_shape_numpy(mean_params: np.ndarray,      # (10,)
                              std_vector: np.ndarray        # (10,)        
                              ) -> np.ndarray:             # (10,)
    """
    Gaussian sampling of shape parameter deviations from the mean.
    """
    shape = mean_params + np.random.randn(mean_params.shape[0]) * std_vector
    return shape


def normal_sample_style_numpy(
        num_garment_classes: int,     
        mean_params: np.ndarray,      # (10,)
        std_vector: np.ndarray        # (10,)
    ) -> np.ndarray:                  # (num_garment_classes, 10)
    """
    Normal sampling of style parameter deviations from the mean, for each garment.
    """
    style = mean_params + np.random.randn(num_garment_classes, mean_params.shape[0]) * std_vector
    return style


def uniform_random_unit_vector(num_vectors):
    """
    Uniform sampling random 3D unit-vectors, i.e. points on unit sphere.
    """
    e = torch.randn(num_vectors, 3)
    e = torch.div(e, torch.norm(e, dim=-1, keepdim=True))
    return e  # (num_vectors, 3)

