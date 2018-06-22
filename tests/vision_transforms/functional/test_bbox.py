
import pytest

import numpy as np

from vision_transforms.functional import bbox as B

seed = 12


def test_check_type():

    with pytest.raises(TypeError):
        B.check_type(None)

    with pytest.raises(TypeError):
        B.check_type([1, 2, 3])

    with pytest.raises(TypeError):
        B.check_type(np.array([[10, 9, 8, 7], ]))

    with pytest.raises(TypeError):
        B.check_type(np.array([[10, 9, 12, 13], [4, 3, 2, 1]]))


def test_crop_multiple():

    input_bbox = np.array([
        [12, 23, 34, 45],
        [56, 67, 78, 89],
    ])

    cropped_bbox = B.crop(input_bbox, 5, 5, 70, 80)
    B.check_type(cropped_bbox)
    true_cropped_bbox = np.array([
        [7, 18, 29, 40],
        [51, 62, 70, 80]
    ])
    assert np.all(cropped_bbox == true_cropped_bbox)


def test_crop_single_bbox():

    input_bbox = np.array([[12, 23, 34, 45], ])

    # bbox is completely in output canvas
    cropped_bbox = B.crop(input_bbox, 0, 0, 50, 50)
    B.check_type(cropped_bbox)
    true_cropped_bbox = np.array([12, 23, 34, 45])
    assert np.all(cropped_bbox == true_cropped_bbox)

    # output canvas is completely inside of bbox
    cropped_bbox = B.crop(input_bbox, 15, 25, 5, 5)
    B.check_type(cropped_bbox)
    true_cropped_bbox = np.array([0, 0, 5, 5])
    assert np.all(cropped_bbox == true_cropped_bbox)

    # bbox is completely outside the output canvas
    cropped_bbox = B.crop(input_bbox, 50, 60, 50, 50)
    B.check_type(cropped_bbox)
    true_cropped_bbox = np.array([0, 0, 0, 0])
    assert np.all(cropped_bbox == true_cropped_bbox)

    # output canvas intersects bbox
    # BR inside bbox
    cropped_bbox = B.crop(input_bbox, 0, 0, 20, 30)
    B.check_type(cropped_bbox)
    true_cropped_bbox = np.array([12, 23, 20, 30])
    assert np.all(cropped_bbox == true_cropped_bbox)

    # TL inside bbox
    cropped_bbox = B.crop(input_bbox, 20, 30, 50, 50)
    true_cropped_bbox = np.array([0, 0, 14, 15])
    assert np.all(cropped_bbox == true_cropped_bbox)

    # BL inside bbox
    cropped_bbox = B.crop(input_bbox, 20, 10, 50, 50)
    true_cropped_bbox = np.array([0, 13, 14, 35])
    assert np.all(cropped_bbox == true_cropped_bbox)

    # TR inside bbox
    cropped_bbox = B.crop(input_bbox, 10, 30, 20, 50)
    true_cropped_bbox = np.array([2, 0, 20, 15])
    assert np.all(cropped_bbox == true_cropped_bbox)


def test_pad():

    input_bbox = np.array([
        [12, 23, 34, 45],
        [56, 67, 78, 89],
    ])
    padding = 5
    true_padded_bbox = input_bbox.copy()
    true_padded_bbox[:, :] += padding

    padded_bbox = B.pad(input_bbox, padding=padding)
    B.check_type(padded_bbox)
    assert np.all(padded_bbox == true_padded_bbox)
    assert id(padded_bbox) == id(input_bbox)
    assert np.all(np.abs(true_padded_bbox[:, 0] - true_padded_bbox[:, 2]) ==
                  np.abs(padded_bbox[:, 0] - padded_bbox[:, 2]))
    assert np.all(np.abs(true_padded_bbox[:, 1] - true_padded_bbox[:, 3]) ==
                  np.abs(padded_bbox[:, 1] - padded_bbox[:, 3]))

    padded_bbox = B.pad(input_bbox, padding=padding, inplace=False)
    assert id(padded_bbox) != id(input_bbox)

    padding = (5, 7)
    true_padded_bbox = input_bbox.copy()
    true_padded_bbox[:, :2] += (padding[0], padding[1])
    true_padded_bbox[:, 2:] += (padding[0], padding[1])

    padded_bbox = B.pad(input_bbox, padding=padding)
    B.check_type(padded_bbox)
    assert np.all(padded_bbox == true_padded_bbox)

    padding = (4, 5, 7, 8)
    true_padded_bbox = input_bbox.copy()
    true_padded_bbox[:, 0] += padding[0]
    true_padded_bbox[:, 1] += padding[1]
    true_padded_bbox[:, 2] += padding[0]
    true_padded_bbox[:, 3] += padding[1]

    padded_bbox = B.pad(input_bbox, padding=padding)
    B.check_type(padded_bbox)
    assert np.all(padded_bbox == true_padded_bbox)
