from typing import List, Tuple
import numpy as np


def concatenate_meshes(vertices_list: List[np.ndarray],     # N[(V, 3)]
                       faces_list: List[np.ndarray]         # N[(F, 3)]
                       ) -> Tuple[np.ndarray, np.ndarray]:  # (N*V, 3), (N*F, 3)
    '''Concatenate a list of vertices arrays and a list of faces arrays.'''

    concat_vertices = np.concatenate(vertices_list, axis=0)

    vertices_list_lengths = [x.shape[0] for x in vertices_list]    
    assert(concat_vertices.shape[1] == 3 and \
        concat_vertices.shape[0] == sum(vertices_list_lengths))

    faces_list_with_offsets = [faces_list[0]]
    for idx in range(len(faces_list) - 1):
        faces_list_with_offsets.append(faces_list[idx + 1] + faces_list[idx].max())
    concat_faces = np.concatenate(faces_list_with_offsets, axis=0)

    faces_list_lengths = [x.shape[0] for x in faces_list]
    assert(concat_faces.shape[1] == 3 and \
        concat_faces.max() > faces_list[0].max() and \
        concat_faces.shape[0] == sum(faces_list_lengths))

    return concat_vertices, concat_faces
