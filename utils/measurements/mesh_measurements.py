from typing import Optional
from math import nan
import numpy as np
import trimesh
import sys

from smplx.body_models import SMPL

sys.path.append('/garmentor/utils/measurements/')

from utils.measurements.model import set_shape
from _utils import get_dist, get_segment_length, get_height


HORIZ_NORMAL = np.array([0, 1, 0], dtype=np.float32)
VERT_NORMAL = np.array([1, 0, 0], dtype=np.float32)


class MeshMeasurements:

    # Mesh landmark indexes.
    HEAD_TOP = 412
    LEFT_HEEL = 3463
    LEFT_NIPPLE = 598
    BELLY_BUTTON = 3500
    INSEAM_POINT = 3149
    LEFT_SHOULDER = 3011
    RIGHT_SHOULDER = 6470
    LEFT_CHEST = 1423
    RIGHT_CHEST = 4896
    LEFT_WAIST = 631
    RIGHT_WAIST = 4424
    UPPER_BELLY_POINT = 3504
    REVERSE_BELLY_POINT = 3502
    LEFT_HIP = 1229
    RIGHT_HIP = 4949
    LEFT_MID_FINGER = 2445
    RIGHT_MID_FINGER = 5906
    LEFT_WRIST = 2241
    RIGHT_WRIST = 5702
    LEFT_INNER_ELBOW = 1663
    RIGHT_INNER_ELBOW = 5121

    SHOULDER_TOP = 3068
    LOW_LEFT_HIP = 3134
    LEFT_ANKLE = 3334

    LOWER_BELLY_POINT = 1769
    #FOREHEAD_POINT = 335
    FOREHEAD_POINT = 336
    #NECK_POINT = 3839
    NECK_POINT = 3049
    #PELVIS_POINT = 3145
    HIP_POINT = 1806
    RIGHT_BICEP_POINT = 6281
    RIGHT_FOREARM_POINT = 5084
    RIGHT_THIGH_POINT = 4971
    RIGHT_CALF_POINT = 4589
    RIGHT_ANKLE_POINT = 6723

    # Mesh measurement idnexes.
    OVERALL_HEIGHT = (HEAD_TOP, LEFT_HEEL)
    SHOULDER_TO_CROTCH_HEIGHT = (SHOULDER_TOP, INSEAM_POINT)
    NIPPLE_HEIGHT = (LEFT_NIPPLE, LEFT_HEEL)
    NAVEL_HEIGHT = (BELLY_BUTTON, LEFT_HEEL)
    INSEAM_HEIGHT = (INSEAM_POINT, LEFT_HEEL)

    SHOULDER_BREADTH = (LEFT_SHOULDER, RIGHT_SHOULDER)
    CHEST_WIDTH = (LEFT_CHEST, RIGHT_CHEST)
    WAIST_WIDTH = (LEFT_WAIST, RIGHT_WAIST)
    TORSO_DEPTH = (UPPER_BELLY_POINT, REVERSE_BELLY_POINT)
    HIP_WIDTH = (LEFT_HIP, RIGHT_HIP)

    ARM_SPAN_FINGERS = (LEFT_MID_FINGER, RIGHT_MID_FINGER)
    ARM_SPAN_WRIST = (LEFT_WRIST, RIGHT_WRIST)
    ARM_LENGTH = (LEFT_SHOULDER, LEFT_WRIST)
    FOREARM_LENGTH = (LEFT_INNER_ELBOW, LEFT_WRIST)
    INSIDE_LEG_LENGTH = (LOW_LEFT_HIP, LEFT_ANKLE)

    # Segmented circumference indices.
    WAIST_INDICES = (3500, 1336, 917, 916, 919, 918, 665, 662, 657, 654, 631, 632, 720, 799, 796, 890, 889, 3124, 3018, \
        3019, 3502, 6473, 6474, 6545, 6376, 6375, 4284, 4285, 4208, 4120, 4121, 4142, 4143, 4150, 4151, 4406, 4405, \
        4403, 4402, 4812)
    CHEST_INDICES = (3076, 2870, 1254, 1255, 1349, 1351, 3033, 3030, 3037, 3034, 3039, 611, 2868, 2864, 2866, 1760, 1419, 741, \
        738, 759, 2957, 2907, 1435, 1436, 1437, 1252, 1235, 749, 752, 3015)     # X2
    WRIST_INDICES = (5573, 5570, 5572, 5564, 5563, 5565, 5566, 5609, 5608, 5568, 5567, 5668, 5669, 5702, 5696, 5691)
    FOREARM_INDICES = (5132, 5133, 5036, 5035, 5192, 5097, 5096, 5113, 5112, 5168, 5171, 5207, 5087, 5083, 5082, 5163)
    BICEP_INDICES = (6282, 4881, 4878, 6281, 6280, 4882, 4883, 4854, 4853, 4278, 4279, 4886, 4744, 4741, 5365, 5010, 5011, 6283)
    HIP_INDICES = (1807, 864, 863, 1205, 1204, 1450, 1799, 868, 867, 937, 816, 815, 1789, 1786, 3111, 3113, 3112, 842, 841, 3158)   # X2
    NECK_INDICES = (3049, 333, 308, 309, 296, 174, 175, 299, 224, 223, 300, 301, 305, 302)     # X2

    # For proposer labels.
    overall_height = None
    
    # Allowable errors.
    AEs = np.array([4.0, nan, nan, nan, 6.0, 5.0, 12.0, nan, 6.0, nan, 5.0, 12.0, nan, nan, 4.0, 
           nan, 6.0, nan, nan, 8.0, 15.0, 6.0, nan, 12.0, nan, 5.0]) * 0.001

    @staticmethod
    def __init_from_betas__(
            smpl_model: SMPL, 
            betas: np.ndarray, 
            mesh_size: Optional[float] = None
        ):
        model_output = set_shape(smpl_model, betas)
        verts = model_output.vertices.detach().cpu().numpy().squeeze()
        faces = smpl_model.faces.squeeze()
        return MeshMeasurements(
            verts=verts, 
            faces=faces, 
            mesh_size=mesh_size
        )

    def __init__(
            self, 
            verts: np.ndarray, 
            faces: np.ndarray, 
            mesh_size: Optional[float] = None
        ):
        self.verts = verts
        self.faces = faces
        self.mesh = trimesh.Trimesh(vertices=self.verts, faces=self.faces)
        self.volume = self.mesh.volume
        self.overall_height = self._get_overall_height()
        if mesh_size is not None:
            self._scale_mesh(mesh_size)

    def _scale_mesh(self, mesh_size):
        self.verts *= (mesh_size / self.overall_height)
        self.overall_height = self._get_overall_height()
        self.mesh = trimesh.Trimesh(vertices=self.verts, faces=self.faces)

    # Use this to obtain overall height, but use overall_height property on the outside.
    def _get_overall_height(self):
        return get_height(
            self.verts[self.OVERALL_HEIGHT[0]], 
            self.verts[self.OVERALL_HEIGHT[1]]
        )

    @property
    def shoulder_to_crotch(self):
        return get_height(
            self.verts[self.SHOULDER_TO_CROTCH_HEIGHT[0]],
            self.verts[self.SHOULDER_TO_CROTCH_HEIGHT[1]]
        )

    @property
    def nipple_height(self):
        return get_height(
            self.verts[self.NIPPLE_HEIGHT[0]], 
            self.verts[self.NIPPLE_HEIGHT[1]]
        )

    @property
    def navel_height(self):
        return get_height(
            self.verts[self.NAVEL_HEIGHT[0]], 
            self.verts[self.NAVEL_HEIGHT[1]]
        )

    @property
    def inseam_height(self):
        return get_height(
            self.verts[self.INSEAM_HEIGHT[0]], 
            self.verts[self.INSEAM_HEIGHT[1]]
        )

    @property
    def shoulder_breadth(self):
        return get_dist([
            self.verts[self.SHOULDER_BREADTH[0]],
            self.verts[self.SHOULDER_BREADTH[1]]
        ])

    @property
    def chest_width(self):
        return get_dist([
            self.verts[self.CHEST_WIDTH[0]],
            self.verts[self.CHEST_WIDTH[1]]
        ])

    @property
    def waist_width(self):
        return get_dist([
            self.verts[self.WAIST_WIDTH[0]],
            self.verts[self.WAIST_WIDTH[1]]
        ])

    @property
    def torso_depth(self):
        return get_dist([
            self.verts[self.TORSO_DEPTH[0]],
            self.verts[self.TORSO_DEPTH[1]]
        ])

    @property
    def hip_width(self):
        return get_dist([
            self.verts[self.HIP_WIDTH[0]],
            self.verts[self.HIP_WIDTH[1]]
        ])

    @property
    def arm_span_fingers(self):
        return get_dist([
            self.verts[self.ARM_SPAN_FINGERS[0]],
            self.verts[self.ARM_SPAN_FINGERS[1]]
        ])

    @property
    def arm_span_wrist(self):
        return get_dist([
            self.verts[self.ARM_SPAN_WRIST[0]],
            self.verts[self.ARM_SPAN_WRIST[1]]
        ])

    @property
    def arm_length(self):
        return get_dist([
            self.verts[self.ARM_LENGTH[0]],
            self.verts[self.ARM_LENGTH[1]]
        ])

    @property
    def forearm_length(self):
        return get_dist([
            self.verts[self.FOREARM_LENGTH[0]],
            self.verts[self.FOREARM_LENGTH[1]]
        ])

    @property
    def inside_leg_length(self):
        return get_height(
            self.verts[self.INSIDE_LEG_LENGTH[0]],
            self.verts[self.INSIDE_LEG_LENGTH[1]]
        )

    @property
    def waist_circumference(self):
        intersection_segments = trimesh.intersections.mesh_plane(
            self.mesh,
            HORIZ_NORMAL,
            self.verts[self.LOWER_BELLY_POINT])
        indexes = self.WAIST_INDICES
        line_segments = np.array([(self.verts[indexes[idx]], self.verts[indexes[idx+1]]) for idx in range(len(indexes) - 1)])
        #return sum([get_dist([x[0], x[1]]) for x in line_segments])
        #vs = [self.verts[idx] for idx in indexes]
        return get_segment_length(intersection_segments)

    @property
    def head_circumference(self):
        intersection_segments = trimesh.intersections.mesh_plane(
            self.mesh,
            HORIZ_NORMAL,
            self.verts[self.FOREHEAD_POINT])
        #return sum([get_dist([x[0], x[1]]) for x in line_segments])
        return get_segment_length(intersection_segments)

    @property
    def neck_circumference(self):
        intersection_segments = trimesh.intersections.mesh_plane(
            self.mesh,
            HORIZ_NORMAL,
            self.verts[self.NECK_POINT])
        indexes = self.NECK_INDICES
        line_segments = np.array([(self.verts[indexes[idx]], self.verts[indexes[idx+1]]) for idx in range(len(indexes) - 1)])
        #return sum([get_dist(x[0], x[1]) for x in line_segments]) * 2.
        vs = [self.verts[idx] for idx in indexes]
        #return get_dist(vs) * 2
        #return sum([get_dist(x[0], x[1]) for x in line_segments])
        return get_segment_length(intersection_segments)

    @property
    def chest_circumference(self):
        intersection_segments = trimesh.intersections.mesh_plane(
            self.mesh,
            HORIZ_NORMAL,
            self.verts[self.LEFT_CHEST])
        indexes = self.CHEST_INDICES
        #line_segments = [(self.verts[indexes[idx]], self.verts[indexes[idx+1]]) for idx in range(len(indexes) - 1)]
        #return sum([get_dist(x[0], x[1]) for x in line_segments]) * 2
        vs = [self.verts[idx] for idx in indexes]
        #return get_dist(vs) * 2
        #return sum([get_dist(x[0], x[1]) for x in line_segments])
        return get_segment_length(intersection_segments)

    @property
    def hip_circumference(self):
        intersection_segments = trimesh.intersections.mesh_plane(
            self.mesh,
            HORIZ_NORMAL,
            self.verts[self.HIP_POINT])
        indexes = self.HIP_INDICES
        #line_segments = [(self.verts[indexes[idx]], self.verts[indexes[idx+1]]) for idx in range(len(indexes) - 1)]
        #return sum([get_dist(x[0], x[1]) for x in line_segments]) * 2
        vs = [self.verts[idx] for idx in indexes]
        #return get_dist(vs) * 2
        #return sum([get_dist([x[0], x[1]]) for x in line_segments])
        return get_segment_length(intersection_segments)

    @property
    def wrist_circumference(self):
        intersection_segments = trimesh.intersections.mesh_plane(
            self.mesh,
            VERT_NORMAL,
            self.verts[self.LEFT_WRIST])
        #indexes = self.WRIST_INDICES
        #line_segments = [(self.verts[indexes[idx]], self.verts[indexes[idx+1]]) for idx in range(len(indexes) - 1)]
        #return sum([get_dist([x[0], x[1]]) for x in line_segments])
        return get_segment_length(intersection_segments)

    @property
    def bicep_circumference(self):
        intersection_segments = trimesh.intersections.mesh_plane(
            self.mesh,
            VERT_NORMAL,
            self.verts[self.RIGHT_BICEP_POINT])
        #indexes = self.BICEP_INDICES
        #line_segments = [(self.verts[indexes[idx]], self.verts[indexes[idx+1]]) for idx in range(len(indexes) - 1)]
        #return sum([get_dist([x[0], x[1]]) for x in line_segments])
        return get_segment_length(intersection_segments)

    @property
    def forearm_circumference(self):
        intersection_segments = trimesh.intersections.mesh_plane(
            self.mesh,
            VERT_NORMAL,
            self.verts[self.RIGHT_FOREARM_POINT])
        #indexes = self.FOREARM_INDICES
        #line_segments = [(self.verts[indexes[idx]], self.verts[indexes[idx+1]]) for idx in range(len(indexes) - 1)]
        #return sum([get_dist([x[0], x[1]]) for x in line_segments])
        return get_segment_length(intersection_segments)

    @property
    def thigh_circumference(self):
        intersection_segments = trimesh.intersections.mesh_plane(
            self.mesh,
            HORIZ_NORMAL,
            self.verts[self.RIGHT_THIGH_POINT])
        #return sum([get_dist([x[0], x[1]]) for x in line_segments]) / 2.
        return get_segment_length(intersection_segments) / 2.

    @property
    def calf_circumference(self):
        intersection_segments = trimesh.intersections.mesh_plane(
            self.mesh,
            HORIZ_NORMAL,
            self.verts[self.RIGHT_CALF_POINT])
        #return sum([get_dist([x[0], x[1]]) for x in line_segments]) / 2.
        return get_segment_length(intersection_segments) / 2.

    @property
    def ankle_circumference(self):
        intersection_segments = trimesh.intersections.mesh_plane(
            self.mesh,
            HORIZ_NORMAL,
            self.verts[self.RIGHT_ANKLE_POINT])
        #return sum([get_dist([x[0], x[1]]) for x in line_segments]) / 2.
        return get_segment_length(intersection_segments) / 2.

    def _get_all_measurements(self):
        return np.array([getattr(self, x) for x in dir(self) if '_' in x and x[0].islower()])

    def _get_ap_measurements(self):
        return np.array([getattr(self, x) for x in MeshMeasurements.aplabels()])

    @property
    def apmeasurements(self):
        return np.array([getattr(self, x) for x in MeshMeasurements.aplabels()])

    @staticmethod
    def alllabels():
        return [x for x in dir(MeshMeasurements) if '_' in x and x[0].islower()]

    @staticmethod
    def aplabels():
        return [
            'head_circumference',
            'neck_circumference',
            'shoulder_to_crotch',
            'chest_circumference',
            'waist_circumference',
            'hip_circumference',
            'wrist_circumference',
            'bicep_circumference',
            'forearm_circumference',
            'arm_length',
            'inside_leg_length',
            'thigh_circumference',
            'calf_circumference',
            'ankle_circumference',
            'shoulder_breadth'
        ]

    @staticmethod
    def letterlabels():
        return [
            'A',
            'B',
            'C',
            'D',
            'E',
            'F',
            'G',
            'H',
            'I',
            'J',
            'K',
            'L',
            'M',
            'N',
            'O'
        ]



def get_measurements(verts, faces):
    return MeshMeasurements(verts, faces).apmeasurements


def get_canonical_volume(
        smpl_model: SMPL,
        betas: np.ndarray, 
        fixed_height: float
    ) -> float:
    return MeshMeasurements.__init_from_betas__(
        smpl_model=smpl_model,
        betas=betas,
        mesh_size=fixed_height
    ).volume
