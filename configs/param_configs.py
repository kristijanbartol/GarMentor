from typing import Dict, List


POSE_CONFIGS = {
    0: {
        'strategy': 'zero',
        'interval': 'intra'
    },
    1: {
        'strategy': 'simple',
        'interval': 'extra'     # currently the same as 'intra'
    },
    2: {
        'strategy': 'simple',
        'interval': 'intra'
    },
    3: {
        'strategy': 'simple',
        'interval': 'extra'     # currently the same as 'intra'
    },
    4: {
        'strategy': 'mocap',
        'interval': 'intra'
    },
    5: {
        'strategy': 'mocap',
        'interval': 'extra'     # currently the same as 'intra'
    },
}

GLOBAL_ORIENT_CONFIGS = {
    0: {
        'strategy': 'zero',
        'interval': 'intra'
    },
    1: {
        'strategy': 'zero',
        'interval': 'extra'
    },
    2: {
        'strategy': 'frontal',
        'interval': 'intra'
    },
    3: {
        'strategy': 'frontal',
        'interval': 'extra'
    },
    4: {
        'strategy': 'diverse',
        'interval': 'intra'
    },
    5: {
        'strategy': 'diverse',
        'interval': 'extra'
    },
    6: {
        'strategy': 'mocap',
        'interval': 'intra'
    },
    7: {
        'strategy': 'mocap',
        'interval': 'extra'
    }
}

SHAPE_CONFIGS = {
    0: {
        'strategy': 'normal',
        'interval': 'intra'
    },
    1: {
        'strategy': 'normal',
        'interval': 'extra'
    },
    2: {
        'strategy': 'uniform',
        'interval': 'intra'
    },
    3: {
        'strategy': 'uniform',
        'interval': 'extra'
    }
}

STYLE_CONFIGS = {
    0: {
        'strategy': 'normal',
        'interval': 'intra'
    },
    1: {
        'strategy': 'normal',
        'interval': 'extra'
    },
    2: {
        'strategy': 'uniform',
        'interval': 'intra'
    },
    3: {
        'strategy': 'uniform',
        'interval': 'extra'
    }
}


def create_param_cfg_dict(cfg_labels: List[int]) -> Dict:
    return {
        'pose': POSE_CONFIGS[cfg_labels[0]],
        'global_orient': GLOBAL_ORIENT_CONFIGS[cfg_labels[1]],
        'shape': SHAPE_CONFIGS[cfg_labels[2]],
        'style': STYLE_CONFIGS[cfg_labels[3]]
    }


TRAIN_TEMPLATE = {
    0: {
        'epochs': [0],      # NOTE: For now, all templates will be fixed throughout the training, therefore, 'epochs': [0]
        'cfgs': [
            [0, 0, 0, 0]    # all intra
        ]
    },
    1: {
        'epochs': [0],
        'cfgs': [
            [0, 1, 1, 1]    # "all" extra
        ]
    }
}
