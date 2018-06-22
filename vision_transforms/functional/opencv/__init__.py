import numbers
import collections

import numpy as np

import cv2

from vision_transforms.functional import _get_affine_matrix33


backend_name = "opencv"


def check_type(img):
    """Check input image type. Raises TypeError if input is incorrect
    """
    if not (isinstance(img, np.ndarray) and img.ndim == 3 and
        img.shape[-1] < img.shape[0] and img.shape[-1] < img.shape[1]):
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


def pad(img, padding, fill=None, **kwargs):
    """Method to pad PIL Image on all sides with constant `fill` value.

    Args:
        img (numpy.ndarray): input image to be padded
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (int, optional): argument for compatibility with 'pillow' backend, if specified it is translated to
            `constant_values`.
        **kwargs: kwargs of `numpy.pad` method, except `pad_width`. For example, `mode` for padding mode,
            `constant_values` to fill with a value if padding mode is constant, etc.

    Returns:
        numpy.ndarray
    """
    check_type(img)

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

    if fill is not None:
        assert 'constant_values' not in kwargs, \
            "Only one argument of `fill` and `constant_values` should be specified"
        kwargs['constant_values'] = fill

    img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), **kwargs)
    return img


def affine(img, angle, translate, scale, shear, resample=0, fillcolor=0, border_mode=0):
    """Apply affine transformation on the image keeping image center invariant

    Args:
        img (numpy.ndarray): input image to be transformed
        angle (float or int): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float): shear angle value in degrees between -180 to 180, clockwise direction.
        resample (int, optional): optional interpolation cv2 flag: `cv2.INTER_NEAREST`, `cv2.INTER_BICUBIC`, etc
            See cv::InterpolationFlags for more details.
        fillcolor (int, optional): optional fill color for the area outside the transform in the output image in case of
            constant border mode. Default is zero pixel
        border_mode (int, optional): pixel extrapolation method (see cv::BorderTypes). Default is constant mode.

    Returns:
        numpy.ndarray
    """
    check_type(img)

    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"

    assert scale > 0.0, "Argument scale should be positive"

    ih, iw, _ = img.shape
    output_size = (iw, ih)
    center = (iw * 0.5 + 0.5, ih * 0.5 + 0.5)
    matrix33 = _get_affine_matrix33(center, angle, translate, scale, shear)
    out_img = cv2.warpAffine(img, matrix33[:2, :], output_size, flags=resample,
                             borderMode=border_mode, borderValue=fillcolor)
    return out_img


def _blend(degenerate, img, factor):
    result = cv2.addWeighted(img, factor, degenerate, (1.0 - factor), 0.0, dtype=-1)
    return result


def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.

    Args:
        img (numpy.ndarray): input image to be adjusted
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        numpy.ndarray
    """
    check_type(img)

    degenerate = np.zeros_like(img)
    return _blend(degenerate, img, brightness_factor)


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.

    Args:
        img (numpy.ndarray): input image to be adjusted. Image should be in RGB mode as we compute gray scale of it.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        numpy.ndarray
    """
    check_type(img)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mean = int(gray.mean() + 0.5)
    degenerate = np.ones_like(img) * mean
    return _blend(degenerate, img, contrast_factor)


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.

    Args:
        img (numpy.ndarray): input image to be adjusted. Image should be in RGB mode as we compute gray scale of it.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        numpy.ndarray
    """
    check_type(img)
    c = img.shape[-1]
    gray = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), axis=-1)
    gray = cv2.merge([gray for _ in range(c)])
    return _blend(gray, img, saturation_factor)


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See `Hue`_ for more details.

    .. _Hue: https://en.wikipedia.org/wiki/Hue

    Args:
        img (numpy.ndarray): input image to be adjusted. Image should be in RGB mode as we compute hue channel of it.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        numpy.ndarray
    """
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    check_type(img)

    # check input according to opencv cvtColor
    # if uint8 -> Hue is between [0, 180]
    # if float32 and range=[0.0, 1.0] -> Hue is between [0, 360]
    # otherwise raise TypeError

    uint8_type = img.dtype == np.uint8
    float32_type = img.dtype == np.float32 and np.max(img) <= 1.0 and np.min(img) >= 0.0

    assert uint8_type or float32_type, \
        "Input image should be uint8 or float32 with range [0.0, 1.0]"

    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    if uint8_type:
        hsv_img[:, :, 0] *= 2.0
        hsv_img[:, :, 1:] /= 255.0

    hsv_img[:, :, 0] = (hsv_img[:, :, 0] + (hue_factor + 0.5) * 360.0) % 360.0

    if uint8_type:
        hsv_img[:, :, 0] /= 2.0
        hsv_img[:, :, 1:] *= 255.0
        hsv_img = hsv_img.astype(np.uint8)

    out_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return out_img
