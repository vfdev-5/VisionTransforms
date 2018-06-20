
import pytest

import numpy as np

from vision_transforms import set_image_backend


seed = 12


@pytest.fixture
def F():
    set_image_backend("opencv")
    from vision_transforms.functional import image as F
    return F


def test_check_type(F):

    with pytest.raises(TypeError):
        F.check_type(None)

    with pytest.raises(TypeError):
        F.check_type([1, 2, 3])

    with pytest.raises(TypeError):
        F.check_type(np.random.rand(100, 120, 3, 4).astype(np.float32))


def test_shape(F):

    np.random.seed(seed)
    img = np.random.rand(100, 120, 3)
    assert F.shape(img) == img.shape


def test_crop(F):

    np.random.seed(seed)
    img = np.random.rand(100, 120, 3)

    x, y, h, w = 10, 11, 12, 13
    cropped_img = img[y:y + h, x:x + w]
    assert np.all(F.crop(img, x, y, w, h) == cropped_img)

