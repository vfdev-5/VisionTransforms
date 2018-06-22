import math

import numpy as np


image = None


def _get_inverse_affine_matrix(center, angle, translate, scale, shear):
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, scale, shear) = [ cos(a)*scale    -sin(a + shear)*scale     0]
    #                              [ sin(a)*scale    cos(a + shear)*scale     0]
    #                              [     0                  0          1]
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    angle = math.radians(angle)
    shear = math.radians(shear)
    scale = 1.0 / scale

    # Inverted rotation matrix with scale and shear
    d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
    matrix = [
        math.cos(angle + shear), math.sin(angle + shear), 0,
        -math.sin(angle), math.cos(angle), 0
    ]
    matrix = [scale / d * m for m in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
    matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += center[0]
    matrix[5] += center[1]
    return matrix


def _get_affine_matrix33(center, angle, translate, scale, shear):

    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
    matrix33 = np.vstack([np.array(matrix).reshape((2, 3)), np.array([0.0, 0.0, 1.0])])
    matrix33 = np.linalg.inv(matrix33)
    return matrix33
