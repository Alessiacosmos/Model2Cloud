"""
@File           : mdl_utils.py
@Author         : Gefei Kong
@Time:          : 25.01.2024 17:58
------------------------------------------------------------------------------------------------------------------------
@Description    : as below

"""

import numpy as np
from .MinimumBoundingBox import MinimumBoundingBox

def calc_face_normal(vertices_f):
    """
    using plane fitting to get the normal
    reference link: https://stackoverflow.com/questions/64818203/how-to-find-the-coordinate-of-points-projection-on-a-planar-surface/64835893#64835893
    :param vertices_f:
    :return:
    """
    # The adjusted plane crosses the centroid of the point collection
    centroid = np.mean(vertices_f, axis=0)

    # Use SVD to calculate the principal axes of the point collection
    # (eigenvectors) and their relative size (eigenvalues)
    _, values, vectors = np.linalg.svd(vertices_f - centroid)

    # Each singular value is paired with its vector and they are sorted from
    # largest to smallest value.
    # The adjusted plane plane must contain the eigenvectors corresponding to
    # the two largest eigenvalues. If only one eigenvector is different
    # from zero, then points are aligned and they don't define a plane.
    # if values[1] < 1e-6:
    #     raise ValueError("Points are aligned, can't define a plane")

    # So the plane normal is the eigenvector with the smallest eigenvalue
    normal = vectors[2]

    return normal


def get_polygonMBR_min_edge(poly_vs:np.ndarray)->float:
    """
    get a polygon's MBR (minimum bounding box) and extract its edge with min length
    s1: point number <= 2:
            smaller or equal to 2 points to organize this polygon, means its a line and should be removed.
    s2: point number > 2:
            a polygon, calculate its MBR, and find its min length edge.
    :param poly_vs:
    :return:
    """
    min_edge_len = 0
    if poly_vs.shape[0]<=2: # smaller or equal to 2 points to organize this polygon, means its a line and should be removed.
        return min_edge_len

    mbr = MinimumBoundingBox(poly_vs[:,:2])
    min_edge_len = min([mbr.length_parallel, mbr.length_orthogonal])

    return min_edge_len