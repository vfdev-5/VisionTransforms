
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


def test_pad(F):

    np.random.seed(seed)
    img = np.random.randint(0, 255, size=(100, 120, 3), dtype=np.uint8)

    padding = 5
    mode = 'constant'
    fill = 10
    padded_img = F.pad(img, padding, mode=mode, constant_values=fill)
    assert padded_img.shape == (img.shape[0] + 2 * padding,
                                img.shape[1] + 2 * padding,
                                img.shape[2])

    assert np.all(padded_img[:padding, :padding, :] == np.ones((padding, padding, 3)) * fill)
    assert np.all(padded_img[-padding:, -padding:, :] == np.ones((padding, padding, 3)) * fill)

    padding = (5, 6)
    padded_img = F.pad(img, padding, mode=mode, constant_values=fill)
    assert padded_img.shape == (img.shape[0] + 2 * padding[1],
                                img.shape[1] + 2 * padding[0],
                                img.shape[2])

    assert np.all(padded_img[:padding[0], :padding[1], :] == np.ones((padding[0], padding[1], 3)) * fill)
    assert np.all(padded_img[-padding[0]:, -padding[1]:, :] == np.ones((padding[0], padding[1], 3)) * fill)

    padding = (5, 6, 7, 8)
    padded_img = F.pad(img, padding, mode=mode, constant_values=fill)
    assert padded_img.shape == (img.shape[0] + padding[1] + padding[3],
                                img.shape[1] + padding[0] + padding[2],
                                img.shape[2])

    assert np.all(padded_img[:padding[0], :padding[1], :] == np.ones((padding[0], padding[1], 3)) * fill)
    assert np.all(padded_img[-padding[2]:, -padding[3]:, :] == np.ones((padding[2], padding[3], 3)) * fill)

    with pytest.raises(AssertionError):
        F.pad(img, padding, mode=mode, fill=fill, constant_values=fill)


def test_adjust_brightness(F):

    from PIL import Image
    from vision_transforms.functional import pillow as FPillow

    np.random.seed(seed)
    img = np.random.randint(0, 70, size=(310, 310, 3), dtype=np.uint8)
    img[10:150, 34:120] += 127
    img[220:250, 134:180] += 140

    pil_img = Image.fromarray(img)
    for a in np.arange(0.0, 1.05, 0.05):
        pil_res = np.asarray(FPillow.adjust_brightness(pil_img, a))
        cv_res = F.adjust_brightness(img, a)
        # Abs Tolerance : 1 <-> 116 vs 117 pix values
        assert np.allclose(pil_res, cv_res, atol=1), "Failed for {}".format(a)


def test_adjust_contrast(F):

    from PIL import Image
    from vision_transforms.functional import pillow as FPillow

    np.random.seed(seed)
    img = np.random.randint(0, 70, size=(310, 310, 3), dtype=np.uint8)
    img[10:150, 34:120] += 127
    img[220:250, 134:180] += 140

    pil_img = Image.fromarray(img)
    for a in np.arange(0.0, 1.05, 0.05):
        pil_res = np.asarray(FPillow.adjust_contrast(pil_img, a))
        cv_res = F.adjust_contrast(img, a)
        # Abs Tolerance : 1 <-> 116 vs 117 pix values
        assert np.allclose(pil_res, cv_res, atol=1), "Failed for {}".format(a)


def test_adjust_saturation(F):

    from PIL import Image
    from vision_transforms.functional import pillow as FPillow

    np.random.seed(seed)
    img = np.random.randint(0, 70, size=(310, 310, 3), dtype=np.uint8)
    img[10:150, 34:120] += 127
    img[220:250, 134:180] += 140

    pil_img = Image.fromarray(img)
    for a in np.arange(0.0, 1.05, 0.05):
        pil_res = np.asarray(FPillow.adjust_saturation(pil_img, a))
        cv_res = F.adjust_saturation(img, a)
        # Abs Tolerance : 2 <-> 116 vs 118 pix values
        assert np.allclose(pil_res, cv_res, atol=2), "Failed for {}".format(a)


def test_adjust_hue(F):

    from PIL import Image
    from vision_transforms.functional import pillow as FPillow

    np.random.seed(seed)
    img = np.random.randint(0, 70, size=(310, 310, 3), dtype=np.uint8)
    img[10:150, 34:120] += 127
    img[220:250, 134:180] += 140

    pil_img = Image.fromarray(img)
    for a in np.arange(-0.5, 0.5, 0.01):
        pil_res = np.asarray(FPillow.adjust_hue(pil_img, a))
        cv_res = F.adjust_hue(img, a)
        # Abs Tolerance : 2 <-> 116 vs 118 pix values
        assert np.allclose(pil_res, cv_res, atol=2), "Failed for {}".format(a)
