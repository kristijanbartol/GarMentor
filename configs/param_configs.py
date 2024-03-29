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
    },
    2: {
        'epochs': [0],
        'cfgs': [
            [0, 2, 0, 0]    # all intra, frontal orients
        ]
    },
    3: {
        'epochs': [0],
        'cfgs': [
            [0, 3, 1, 1]    # "all" extra, frontal orients
        ]
    },
    4: {
        'epochs': [0],
        'cfgs': [
            [2, 3, 1, 1]    # "all" extra, simple poses, frontal orients
        ]
    },
    5: {
        'epochs': [0],
        'cfgs': [
            [2, 5, 1, 1]    # "all" extra, simple poses, diverse orients
        ]
    },
    6: {
        'epochs': [0],
        'cfgs': [
            [4, 3, 1, 1]    # "all" extra, mocap poses, frontal orients
        ]
    },
    7: {
        'epochs': [0],
        'cfgs': [
            [4, 5, 1, 1]    # "all" extra, mocap poses, frontal orients
        ]
    },
    8: {
        'epochs': [0],
        'cfgs': [
            [4, 7, 1, 1]    # "all" extra, mocap poses, mocap orients
        ]
    }
}


def create_param_cfg_dict(cfg_labels: List[int]) -> Dict:
    return {
        'pose': POSE_CONFIGS[cfg_labels[0]],
        'global_orient': GLOBAL_ORIENT_CONFIGS[cfg_labels[1]],
        'shape': SHAPE_CONFIGS[cfg_labels[2]],
        'style': STYLE_CONFIGS[cfg_labels[3]]
    }


def get_param_cfg_from_label(template_label: int) -> Dict:
    return create_param_cfg_dict(TRAIN_TEMPLATE[template_label]['cfgs'][0])
