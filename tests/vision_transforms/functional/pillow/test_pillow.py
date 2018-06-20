
import pytest

import numpy as np
from PIL import Image

from vision_transforms import set_image_backend


seed = 12


@pytest.fixture
def F():
    set_image_backend("pillow")
    from vision_transforms.functional import image as F
    return F


def test_check_type(F):

    with pytest.raises(TypeError):
        F.check_type(None)

    with pytest.raises(TypeError):
        F.check_type([1, 2, 3])

    with pytest.raises(TypeError):
        F.check_type(np.random.rand(100, 120).astype(np.float32))


def test_shape(F):

    np.random.seed(seed)
    img = Image.fromarray(np.random.rand(100, 120).astype(np.float32))
    true_shape = img.size[::-1] + (len(img.getbands()), )
    out_shape = F.shape(img)
    assert out_shape == true_shape, "{} vs {}".format(out_shape, true_shape)

    img = Image.fromarray(np.random.randint(0, 255, size=(100, 120, 3), dtype=np.uint8))
    true_shape = img.size[::-1] + (len(img.getbands()), )
    out_shape = F.shape(img)
    assert out_shape == true_shape, "{} vs {}".format(out_shape, true_shape)


def test_crop(F):

    np.random.seed(seed)
    img = np.random.randint(0, 255, size=(100, 120, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img)

    x, y, h, w = 10, 11, 12, 13
    true_img = Image.fromarray(img[y:y + h, x:x + w])
    assert F.crop(pil_img, x, y, w, h) == true_img


def test_pad(F):

    np.random.seed(seed)
    img = np.random.randint(0, 255, size=(100, 120, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img)

    padding = 5
    padded_img = F.pad(pil_img, padding)
    assert padded_img.size == (pil_img.size[0] + 2 * padding, pil_img.size[1] + 2 * padding)

