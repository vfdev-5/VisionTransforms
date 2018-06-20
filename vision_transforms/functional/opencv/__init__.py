
import numpy as np

import cv2


backend_name = "opencv"


def check_type(img):
    """Check input image type. Raises TypeError if input is incorrect
    """
    if not (isinstance(img, np.ndarray) and img.ndim == 3):
        raise TypeError("Input image should be numpy.ndarray with shape (H, W, C), "
                        "but given '{}'".format(type(img)))


def shape(img):
    """

    Args:
        img (numpy.ndarray): input image of shape (H, W, C)

    Returns:
        tuple

    """
    check_type(img)
    return img.shape


def crop(img, x, y, w, h):
    """Method to crop image

    Args:
        img (numpy.ndarray): input image to crop
        x (int): left crop box coordinate
        y (int): top crop box coordinate
        w (int): output width
        h (int): output height

    Returns:
        numpy.ndarray
    """
    check_type(img)
    return img[y:y + h, x:x + w, :]
