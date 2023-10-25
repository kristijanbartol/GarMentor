from typing import Tuple, Union, Iterator, Any, Optional, Dict
from dataclasses import dataclass, fields
import numpy as np
import os

#sys.path.append('/GarMentor')

import configs.paths as paths


@dataclass
class PreparedSampleValues:

    """
    A standard dataclass used to handle prepared sample values.

    Note that cam_t was supposed to be deprecated, but it is actually
    required by the SURREAL-based datasets for the keypoint strategy
    which takes ground truth joints and projects them orthographically.
    Then the cam_t is applied on top to properly place 2D keypoints.
    For AGORA-like dataset it's a dummy value and it won't be used in
    the training loop with the AGORA-like data. In case I decide that
    the keypoint strategy (ground truth 3D -> projection -> cam_t) is
    necessarily inferior, I will remove cam_t (kbartol).

    Joints 2D data is common for all the datasets, but it differs
    between SURREAL-like and AGORA-like dataset in a way that for 
    SURREAL the 2D joints are used only if the keypoints are pre-
    extracted using the 2D keypoint detector. In case of AGORA, 2D
    joints are mandatory as they can't be projected afterwards (no
    X, Y, Z camera locations in the training time). However, AGORA
    can also create 2D joints by using 2D keypoint detector.

    Bounding box information, on the other hand, is specific to AGORA-
    like data.
    """

    pose: np.ndarray                        # (72,)
    shape: np.ndarray                       # (10,)
    style_vector: np.ndarray                # (4, 10)
    garment_labels: np.ndarray              # (4,)
    joints_3d: np.ndarray                   # (17, 3)
    joints_conf: np.ndarray                 # (17,)
    joints_2d: Optional[np.ndarray] = None  # (17, 2)
    cam_t: Optional[np.ndarray] = None      # (3,)
    bbox: Optional[np.ndarray] = None       # (2, 2)

    def __getitem__(
            self, 
            key: str
        ) -> np.ndarray:
        return getattr(self, key)

    def get(
            self, 
            key, 
            default=None
        ) -> Union[Any, None]:
        return getattr(self, key, default)

    def __iter__(self) -> Iterator[str]:
        return self.keys()

    def keys(self) -> Iterator[str]:
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self) -> Iterator[Any]:
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self) -> Iterator[Tuple[str, Any]]:
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)


class PreparedValuesArray():

    """
    A class used to keep track of an array of prepared sampled values.
    """

    def __init__(
            self, 
            samples_dict: Optional[Dict[str, np.ndarray]] = None
        ) -> None:
        if samples_dict is not None:
            self._samples_dict = {k: list(v) for k, v in samples_dict.items()}
            self.keys = samples_dict.keys()
        else:
            self._samples_dict = {}
            self.keys = []

    def _set_dict_keys(
            self, 
            sample_dict_keys: Iterator[str]
        ) -> None:
        """
        Add 's' to key name specify plural.
        """
        self.keys = [x + 's' for x in sample_dict_keys]

    def append(
            self, 
            values: PreparedSampleValues
        ) -> None:
        """
        Mimics adding to list of values to an array. Saves to latent dict.
        """
        if not self._samples_dict:
            self._set_dict_keys(values.keys())
            for k in self.keys:
                self._samples_dict[k] = []
        for ks, k in zip(self.keys, values.keys()):
            self._samples_dict[ks].append(values[k])

    def get(self) -> Dict[str, np.ndarray]:
        """
        Get the dictionary with all np.ndarray items.
        """
        return_dict = {k: np.empty(0,) for k, _ in self._samples_dict.items()}
        for ks in self.keys:
            return_dict[ks] = np.array(self._samples_dict[ks])
        return return_dict


def get_dataset_dirs(
            param_cfg: Dict,
            garment_model: str,
            gender: str,
            img_wh: int,
            upper_class: Optional[str] = None,
            lower_class: Optional[str] = None,
    ) -> Dict[str, str]:
    return {data_split: os.path.join(
        paths.DATA_ROOT_DIR,
        garment_model,
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
        f'{upper_class}+{lower_class}' if garment_model == 'tn' else ''
    ) for data_split in ['train', 'valid'] }
