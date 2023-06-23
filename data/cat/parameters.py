from typing import Optional, Tuple
import os
import numpy as np

from configs import paths
from configs.const import SAMPLING_STRATEGIES
from configs.poseMF_shapeGaussian_net_config import get_cfg_defaults
import utils.sampling_utils


class Parameters(object):

    """
    The class for representing and managing (sampling) parameters.
    """

    def __init__(self):
        self.params_dir = os.path.join(
            paths.DATA_ROOT_DIR,
            paths.PARAMS_DIR
        )
        self.cfg = get_cfg_defaults()
        self.sampling_cfg = self.cfg.TRAIN.SYNTH_DATA.SAMPLING

    def get_save_path(
            self,
            param_type: str,
            sampling_strategy: str,
            interval_type: Optional[str] = None
        ) -> str:
        return os.path.join(
            self.params_dir,
            param_type,
            sampling_strategy,
            interval_type if interval_type is not None else 'intra'
        )
    
    def save_params(
            self,
            save_dir: str,
            params: Tuple[np.ndarray, np.ndarray],
        ) -> None:
        for split_idx, data_split in enumerate(['train', 'valid']):
            if os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(
                save_dir,
                f'{data_split}.npy'
            )
            np.save(save_path, params[split_idx])

    def sample_params(
            self,
            param_type: str,
            sampling_strategy: str
        ) -> Tuple[np.ndarray, np.ndarray]:
        sampler_fun = getattr(
            utils.sampling_utils, 
            f'sample_{sampling_strategy}_{param_type}'
        )
        return sampler_fun(
            self.sampling_cfg
        )

    def sample_all_params(self) -> None:
        """
        Sample random pose, global orient, shape, and style parameters.
        """
        for param_type in ['pose', 'global_orient', 'shape', 'style']:
            for sampling_strategy in SAMPLING_STRATEGIES[param_type]:
                save_path = self.get_save_path(
                    param_type=param_type,
                    sampling_strategy=sampling_strategy
                )
                if not os.path.exists(save_path):
                    params = self.sample_params(
                        param_type=param_type,
                        sampling_strategy=sampling_strategy
                    )
                    self.save_params(save_path, params)
