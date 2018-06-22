import math

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
    fill = (10, 11, 12)
    padded_img = F.pad(pil_img, padding, fill)
    assert padded_img.size == (pil_img.size[0] + 2 * padding, pil_img.size[1] + 2 * padding)

    np_padded_img = np.asarray(padded_img)
    assert np.all(np_padded_img[:padding, :padding, :] == np.ones((padding, padding, 3)) * fill)
    assert np.all(np_padded_img[-padding:, -padding:, :] == np.ones((padding, padding, 3)) * fill)

    padding = (5, 6)
    padded_img = F.pad(pil_img, padding, fill)
    assert padded_img.size == (pil_img.size[0] + 2 * padding[0], pil_img.size[1] + 2 * padding[1])

    np_padded_img = np.asarray(padded_img)
    assert np.all(np_padded_img[:padding[0], :padding[1], :] == np.ones((padding[0], padding[1], 3)) * fill)
    assert np.all(np_padded_img[-padding[0]:, -padding[1]:, :] == np.ones((padding[0], padding[1], 3)) * fill)

    padding = (5, 6, 7, 8)
    padded_img = F.pad(pil_img, padding, fill)
    assert padded_img.size == (pil_img.size[0] + padding[0] + padding[2], pil_img.size[1] + padding[1] + padding[3])

    np_padded_img = np.asarray(padded_img)
    assert np.all(np_padded_img[:padding[0], :padding[1], :] == np.ones((padding[0], padding[1], 3)) * fill)
    assert np.all(np_padded_img[-padding[2]:, -padding[3]:, :] == np.ones((padding[2], padding[3], 3)) * fill)


def test_affine(F):
    input_img = np.zeros((200, 200, 3), dtype=np.uint8)
    # pts = []
    cnt = [100, 100]
    for pt in [(80, 80), (100, 80), (100, 100)]:
        for i in range(-5, 5):
            for j in range(-5, 5):
                input_img[pt[0] + i, pt[1] + j, :] = [255, 155, 55]
    #            pts.append((pt[0] + i, pt[1] + j))
    # pts = list(set(pts))

    with pytest.raises(TypeError):
        F.affine(input_img, 10)

    pil_img = Image.fromarray(input_img)

    def _to_3x3_inv(inv_result_matrix):
        result_matrix = np.zeros((3, 3))
        result_matrix[:2, :] = np.array(inv_result_matrix).reshape((2, 3))
        result_matrix[2, 2] = 1
        return np.linalg.inv(result_matrix)

    def _test_transformation(a, t, s, sh):
        a_rad = math.radians(a)
        s_rad = math.radians(sh)
        # 1) Check transformation matrix:
        c_matrix = np.array([[1.0, 0.0, cnt[0]], [0.0, 1.0, cnt[1]], [0.0, 0.0, 1.0]])
        c_inv_matrix = np.linalg.inv(c_matrix)
        t_matrix = np.array([[1.0, 0.0, t[0]],
                             [0.0, 1.0, t[1]],
                             [0.0, 0.0, 1.0]])
        r_matrix = np.array([[s * math.cos(a_rad), -s * math.sin(a_rad + s_rad), 0.0],
                             [s * math.sin(a_rad), s * math.cos(a_rad + s_rad), 0.0],
                             [0.0, 0.0, 1.0]])
        true_matrix = np.dot(t_matrix, np.dot(c_matrix, np.dot(r_matrix, c_inv_matrix)))
        result_matrix = _to_3x3_inv(F._get_inverse_affine_matrix(center=cnt, angle=a,
                                                                 translate=t, scale=s, shear=sh))
        assert np.sum(np.abs(true_matrix - result_matrix)) < 1e-10
        # 2) Perform inverse mapping:
        true_result = np.zeros((200, 200, 3), dtype=np.uint8)
        inv_true_matrix = np.linalg.inv(true_matrix)
        for y in range(true_result.shape[0]):
            for x in range(true_result.shape[1]):
                res = np.dot(inv_true_matrix, [x, y, 1])
                _x = int(res[0] + 0.5)
                _y = int(res[1] + 0.5)
                if 0 <= _x < input_img.shape[1] and 0 <= _y < input_img.shape[0]:
                    true_result[y, x, :] = input_img[_y, _x, :]

        result = F.affine(pil_img, angle=a, translate=t, scale=s, shear=sh)
        assert result.size == pil_img.size
        # Compute number of different pixels:
        np_result = np.array(result)
        n_diff_pixels = np.sum(np_result != true_result) / 3
        # Accept 3 wrong pixels
        assert n_diff_pixels < 3, \
            "a={}, t={}, s={}, sh={}\n".format(a, t, s, sh) + \
            "n diff pixels={}\n".format(np.sum(np.array(result)[:, :, 0] != true_result[:, :, 0]))

    # Test rotation
    a = 45
    _test_transformation(a=a, t=(0, 0), s=1.0, sh=0.0)

    # Test translation
    t = [10, 15]
    _test_transformation(a=0.0, t=t, s=1.0, sh=0.0)

    # Test scale
    s = 1.2
    _test_transformation(a=0.0, t=(0.0, 0.0), s=s, sh=0.0)

    # Test shear
    sh = 45.0
    _test_transformation(a=0.0, t=(0.0, 0.0), s=1.0, sh=sh)

    # # Test rotation, scale, translation, shear
    # for a in range(-90, 90, 25):
    #     for t1 in range(-10, 10, 5):
    #         for s in [0.75, 0.98, 1.0, 1.1, 1.2]:
    #             for sh in range(-15, 15, 5):
    #                 _test_transformation(a=a, t=(t1, t1), s=s, sh=sh)


def test_adjust_brightness(F):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
    x_pil = Image.fromarray(x_np, mode='RGB')

    # test 0
    y_pil = F.adjust_brightness(x_pil, 1)
    y_np = np.array(y_pil)
    assert np.allclose(y_np, x_np)

    # test 1
    y_pil = F.adjust_brightness(x_pil, 0.5)
    y_np = np.array(y_pil)
    y_ans = [0, 2, 6, 27, 67, 113, 18, 4, 117, 45, 127, 0]
    y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
    assert np.allclose(y_np, y_ans)

    # test 2
    y_pil = F.adjust_brightness(x_pil, 2)
    y_np = np.array(y_pil)
    y_ans = [0, 10, 26, 108, 255, 255, 74, 16, 255, 180, 255, 2]
    y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
    assert np.allclose(y_np, y_ans)

