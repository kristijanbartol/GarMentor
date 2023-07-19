from typing import Tuple, Dict, Optional
import os
import numpy as np
import sys

sys.path.append('/GarMentor/')

from configs import paths
from configs.const import SAMPLING_STRATEGIES
from configs.poseMF_shapeGaussian_net_config import get_cfg_defaults
import utils.sampling_utils


class Parameters(object):

    """
    The class for representing and managing (sampling) parameters.
    """

    ALL_PARAMS_KEYS = ['pose', 'global_orient', 'shape', 'style']

    def __init__(
            self,
            param_cfg: Optional[Dict[str, Dict[str, str]]] = None
        ) -> None:
        self.params_dir = os.path.join(
            paths.DATA_ROOT_DIR,
            paths.PARAMS_DIR
        )
        self.cfg = get_cfg_defaults()
        self.sampling_cfg = self.cfg.TRAIN.SYNTH_DATA.SAMPLING
        if param_cfg is not None:
            self.params_dict, self.params_sizes = self.load_params(param_cfg)

    def load_params(
            self,
            param_cfg: Dict[str, Dict[str, str]]
        ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, int]]]:
        params_dict = {x: {'train': np.empty(0), 'valid': np.empty(0)} for x in self.ALL_PARAMS_KEYS}
        params_sizes = {x: {'train': 0, 'valid': 0} for x in self.ALL_PARAMS_KEYS}
        for param_type in self.ALL_PARAMS_KEYS:
            sampling_strategy = param_cfg[param_type]['strategy']
            interval_type = param_cfg[param_type]['interval']
            train_params, valid_params = self.init_params(
                param_type=param_type,
                sampling_strategy=sampling_strategy,
                interval_type=interval_type
            )
            params_dict[param_type]['train'] = train_params
            params_dict[param_type]['valid'] = valid_params
            params_sizes[param_type]['train'] = params_dict[param_type]['train'].shape[0]
            params_sizes[param_type]['valid'] = params_dict[param_type]['valid'].shape[0]
        return params_dict, params_sizes

    def get_path(
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
    
    def _save_params(
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

    def process_style(self) -> Tuple[np.ndarray, np.ndarray]:
        upper_style = getattr(
            utils.sampling_utils, 
            f'sample_predefined_style'
        )(None, 'upper')
        lower_style = getattr(
            utils.sampling_utils, 
            f'sample_predefined_style'
        )(None, 'lower')
        return (
            np.concatenate((upper_style[0][:, None], lower_style[0][:, None]), axis=1),
            np.concatenate((upper_style[1][:, None], lower_style[1][:, None]), axis=1)
        )

    def init_params(
            self,
            param_type: str,
            sampling_strategy: str,
            interval_type: str
        ) -> Tuple[np.ndarray, np.ndarray]:
        params_path = self.get_path(
            param_type=param_type,
            sampling_strategy=sampling_strategy,
            interval_type=interval_type
        )
        cfg_str = f'({param_type}-{sampling_strategy}-{interval_type})'
        if not os.path.exists(params_path):
            print(f'Sampling {cfg_str}...')
            if param_type == 'style':
                params = self.process_style()
            else:
                params = getattr(
                    utils.sampling_utils, 
                    f'sample_{sampling_strategy}_{param_type}'
                )(interval_type)
            self._save_params(params_path, params)
            return params
        else:
            print(f'Loading {cfg_str}...')
            return tuple([np.load(os.path.join(
                    params_path, f'{data_split}.npy')
                ) for data_split in ['train', 'valid']])

    def init_all_params(self) -> None:
        """
        Sample random pose, global orient, shape, and style parameters.
        """
        for param_type in self.ALL_PARAMS_KEYS:
            for sampling_strategy in SAMPLING_STRATEGIES[param_type]:
                for interval_type in ['intra', 'extra']:
                    self.init_params(
                        param_type=param_type,
                        sampling_strategy=sampling_strategy,
                        interval_type=interval_type
                    )

    def get_param_type(
            self,
            data_split: str,
            param_type: str,
            idx: int
        ) -> np.ndarray:
        param_size = self.params_sizes[param_type][data_split]
        return self.params_dict[param_type][data_split][idx % param_size]
    
    @staticmethod
    def _post_prepare(params_sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        params_sample['pose'] = np.concatenate(
            (params_sample['global_orient'], params_sample['pose']), axis=0)
        return params_sample

    def get(self,
            data_split: str,
            idx: int
        ) -> Dict[str, np.ndarray]:
        return self._post_prepare(
            {x: self.get_param_type(
                data_split=data_split,
                param_type=x,
                idx=idx
            ) for x in self.ALL_PARAMS_KEYS})


if __name__ == '__main__':
    parameters = Parameters()
    parameters.init_all_params()
    style = parameters.get_param_type('train', 'style', 0)
