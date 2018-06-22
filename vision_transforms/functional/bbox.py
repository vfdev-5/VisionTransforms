import numbers
import collections

import numpy as np


from vision_transforms.functional import _get_inverse_affine_matrix


def check_type(bboxes):
    """Check input bounding boxes type. Raises TypeError if input is incorrect
    """
    if not (isinstance(bboxes, np.ndarray) and bboxes.ndim == 2 and bboxes.shape[1] == 4):
        raise TypeError("Input bbox should be numpy.ndarray of type [[x1, y1, x2, x2], ...],"
                        "but given '{}'".format(type(bboxes)))
    # Check xyxy type:
    if not (np.all(bboxes[:, 0] <= bboxes[:, 2]) and np.all(bboxes[:, 1] <= bboxes[:, 3])):
        raise TypeError("Input bboxes should be all of type [x1, y1, x2, x2] with x1 <= x2 and y1 <= y2")


def crop(bboxes, x, y, w, h):
    """Method to crop bounding boxes

    Args:
        bboxes (numpy.ndarray): input bbox of type [[x1, y1, x2, y2], ...]
        x (int): left crop box coordinate
        y (int): top crop box coordinate
        w (int): output width
        h (int): output height

    Returns:
        cropped bounding box
    """
    check_type(bboxes)

    pt = np.array([x, y, x, y])
    a_min = np.array([0, 0, 0, 0])
    a_max = np.array([w, h, w, h])
    clipped_bboxes = np.clip(bboxes - pt, a_min=a_min, a_max=a_max, )
    return clipped_bboxes


def _get_ltrb_padding(padding):
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, collections.Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, collections.Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]
    return pad_left, pad_top, pad_right, pad_bottom


def pad(bboxes, padding, inplace=True):
    """Method to pad bounding boxes (inplace)

    Args:
        bboxes (numpy.ndarray): input bbox of type [[x1, y1, x2, y2], ...]
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        inplace (bool, optional): if True input bounding boxes are modified, otherwise they are copied

    Returns:
        padded bounding box
    """
    check_type(bboxes)

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')

    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    pad_left, pad_top, _, _ = _get_ltrb_padding(padding)

    if not inplace:
        bboxes = bboxes.copy()

    bboxes[:, :] += (pad_left, pad_top, pad_left, pad_top)
    return bboxes


def affine(bboxes, center, angle, translate, scale, shear):
    """Apply affine transformation on the bounding boxes keeping a center invariant

    Args:
        bboxes (numpy.ndarray): input bbox of type [[x1, y1, x2, y2], ...]
        center (tuple): canvas center point that remains not transformed
        angle (float or int): Unused as does not make sense. Remains for compatibility with other affine transformations
        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float): Unused as does not make sense. Remains for compatibility with other affine transformations
    """
    check_type(bboxes)

    assert angle == 0 and shear == 0.0, \
        "Angle and shear should be 0.0. It does not make sense to rotate or shear bounding boxes."

    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"

    assert scale > 0.0, "Argument scale should be positive"

    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
    matrix33 = np.vstack([np.array(matrix).reshape((2, 3)), np.array([0.0, 0.0, 1.0])])
    matrix33 = np.linalg.inv(matrix33)

    # bboxes to points
    ll = len(bboxes)
    points = np.ones((3, 2 * ll))
    points[0, :ll] = bboxes[:, 0]
    points[1, :ll] = bboxes[:, 1]
    points[0, ll:] = bboxes[:, 2]
    points[1, ll:] = bboxes[:, 3]

    # Apply transformation
    t_points = np.dot(matrix33, points)

    # Points to bboxes
    t_bboxes = np.zeros_like(bboxes).astype(np.float)
    t_bboxes[:, 0] = t_points[0, :ll]
    t_bboxes[:, 1] = t_points[1, :ll]
    t_bboxes[:, 2] = t_points[0, ll:]
    t_bboxes[:, 3] = t_points[1, ll:]
    return t_bboxes
