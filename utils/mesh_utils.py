import numpy as np


def concatenate_meshes(vertices_list, faces_list):
    concat_vertices = np.concatenate(vertices_list, axis=0)
    faces_list_with_offsets = [faces_list[0]]
    for idx in range(len(faces_list) - 1):
        faces_list_with_offsets = faces_list[idx + 1] + faces_list[idx].max()
    concat_faces = np.concatenate(faces_list_with_offsets, axis=0)
    return concat_vertices, concat_faces
