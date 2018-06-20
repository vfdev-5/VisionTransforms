import numbers
import collections

import numpy as np


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


def pad(bboxes, padding, inplace=True):
    """Method to pad bounding boxes (inplace)

    Args:
        bboxes (numpy.ndarray): input bbox of type [[x1, y1, x2, y2], ...]
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.

    Returns:
        padded bounding box
    """
    check_type(bboxes)

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')

    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

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

    if not inplace:
        bboxes = bboxes.copy()

    bboxes[:, :] += (-pad_left, -pad_top, pad_right, pad_bottom)
    return bboxes
