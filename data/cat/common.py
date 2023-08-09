from typing import Dict
import os

from configs import paths

def get_dataset_dirs(
            param_cfg: Dict,
            upper_class: str,
            lower_class: str,
            gender: str,
            img_wh: int
        ) -> Dict[str, str]:
        return {data_split: os.path.join(
            paths.DATA_ROOT_DIR,
            paths.DATASETS_DIR,
            f"{param_cfg['pose']['strategy']}-pose",
            param_cfg['pose']['interval'],
            f"{param_cfg['global_orient']['strategy']}-global_orient",
            param_cfg['global_orient']['interval'],
            f"shape-{param_cfg['shape']['strategy']}",
            param_cfg['shape']['interval'],
            f"style-{param_cfg['shape']['strategy']}",
            param_cfg['style']['interval'],
            str(img_wh),
            data_split,
            gender,
            f'{upper_class}+{lower_class}'
        ) for data_split in ['train', 'valid'] }
