
import pytest

import numpy as np
from PIL import Image

from vision_transforms import set_image_backend
from vision_transforms.transforms import RandomCrop, BBoxRandomCrop
from vision_transforms.functional import bbox as B


seed = 12


@pytest.fixture
def F():
    set_image_backend("pillow")
    from vision_transforms.functional import image as F
    return F


def test_random_crop(F):

    np.random.seed(seed)
    img = np.random.randint(0, 255, size=(100, 120, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img)

    t = RandomCrop(size=50)
    t_pil_img = t(pil_img)
    assert t_pil_img.size == (50, 50) and t_pil_img.mode == pil_img.mode

    t = RandomCrop(size=50)
    rng = t.get_rng()
    params = t.get_params((100, 120), (50, 50))
    t_pil_img = t(pil_img, rng)
    true_t_pil_img = F.crop(pil_img, *params)
    assert t_pil_img == true_t_pil_img

    t = RandomCrop(size=50)
    t_pil_img2 = t(pil_img, rng)
    assert t_pil_img2 == true_t_pil_img

    t = RandomCrop(size=(50, 50))
    rng = t.get_rng()
    params = t.get_params((100, 120), (50, 50))
    t_pil_img = t(pil_img, rng)
    true_t_pil_img = F.crop(pil_img, *params)
    assert t_pil_img == true_t_pil_img


def test_bbox_random_crop():

    np.random.seed(seed)

    input_canvas = (100, 100)

    x1 = np.random.randint(0, int(0.7 * input_canvas[1]))
    y1 = np.random.randint(0, int(0.7 * input_canvas[0]))
    x2 = np.random.randint(x1, input_canvas[1])
    y2 = np.random.randint(y1, input_canvas[0])

    input_bbox = np.array([[x1, y1, x2, y2], ])

    output_canvas = (50, 50)

    import random
    random.seed(seed)

    t = BBoxRandomCrop(input_canvas_size=input_canvas, size=output_canvas)
    rng = t.get_rng()
    params = t.get_params(input_canvas, output_canvas)
    t_bbox = t(input_bbox, rng)
    B.check_type(t_bbox)
    true_t_bbox = B.crop(input_bbox, *params)
    assert np.all(t_bbox == true_t_bbox)

    t._setup_rng(rng)
    params2 = t.get_params(input_canvas, output_canvas)
    assert params == params2

    t2 = BBoxRandomCrop(input_canvas_size=input_canvas[0], size=output_canvas[0])
    t_bbox2 = t2(input_bbox, rng)
    assert np.all(t_bbox2 == true_t_bbox)
