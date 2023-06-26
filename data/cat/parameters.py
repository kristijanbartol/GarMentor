from typing import Optional, Tuple
import os
import numpy as np
import sys

sys.path.append('/garmentor/')

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
            interval_type: str
        ) -> str:
        return os.path.join(
            self.params_dir,
            param_type,
            sampling_strategy,
            interval_type
        )
    
    def save_params(
            self,
            save_dir: str,
            params: Tuple[np.ndarray, np.ndarray],
        ) -> None:
        for split_idx, data_split in enumerate(['train', 'valid']):
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir,
                f'{data_split}.npy'
            )
            np.save(save_path, params[split_idx])

    def sample_params(
            self,
            param_type: str,
            sampling_strategy: str,
            interval_type: str
        ) -> Tuple[np.ndarray, np.ndarray]:
        return getattr(
            utils.sampling_utils, 
            f'sample_{sampling_strategy}_{param_type}'
        )(interval_type)

    def sample_all_params(self) -> None:
        """
        Sample random pose, global orient, shape, and style parameters.
        """
        for param_type in ['pose', 'global_orient', 'shape', 'style']:
            for sampling_strategy in SAMPLING_STRATEGIES[param_type]:
                for interval_type in ['intra', 'extra']:
                    print(f'Sampling ({param_type}, {sampling_strategy}, {interval_type})...')
                    save_path = self.get_save_path(
                        param_type=param_type,
                        sampling_strategy=sampling_strategy,
                        interval_type=interval_type
                    )
                    if not os.path.exists(save_path):
                        params = self.sample_params(
                            param_type=param_type,
                            sampling_strategy=sampling_strategy,
                            interval_type=interval_type
                        )
                        self.save_params(save_path, params)


if __name__ == '__main__':
    parameters = Parameters()
    parameters.sample_all_params()
