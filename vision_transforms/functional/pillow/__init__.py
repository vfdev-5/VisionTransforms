import numbers
import collections

import numpy as np

from PIL import Image, ImageOps, PILLOW_VERSION, ImageEnhance

from vision_transforms.functional import _get_inverse_affine_matrix


backend_name = "pillow"


def check_type(img):
    if not isinstance(img, Image.Image):
        raise TypeError("Input image should be PIL.Image.Image, "
                        "but given '{}'".format(type(img)))


def shape(img):
    """

    Args:
        img:

    Returns:

    """
    check_type(img)

    w, h = img.size
    c = len(img.getbands())
    return h, w, c


def crop(img, x, y, w, h):
    """Method to crop image

    Args:
        img (PIL.Image.Image): input image to crop
        x (int): left crop box coordinate
        y (int): top crop box coordinate
        w (int): output width
        h (int): output height

    Returns:
        PIL.Image.Image
    """
    check_type(img)
    return img.crop((x, y, x + w, y + h))


def pad(img, padding, fill=0, mode='constant'):
    """Method to pad PIL Image on all sides with constant `fill` value.

    Args:
        img (PIL Image): Image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        mode (str, optional): unused argument for compatibility with 'opencv' backend

    Returns:
        Padded image
    """
    check_type(img)

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')

    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    assert mode in ['constant', ], 'Padding mode should be constant'

    return ImageOps.expand(img, border=padding, fill=fill)


def affine(img, angle, translate, scale, shear, resample=0, fillcolor=None, border_mode=None):
    """Apply affine transformation on the image keeping image center invariant

    Args:
        img (PIL Image): PIL Image to be rotated.
        angle (float or int): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float): shear angle value in degrees between -180 to 180, clockwise direction.
        resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):
            An optional resampling filter.
            See `PIL Image.transform <http://pillow.readthedocs.io/en/5.1.x/reference/Image.html#PIL.Image.Image.transform>`_
            for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
        border_mode (int, optional): unused argument for compatibility with 'opencv' backend
    """
    check_type(img)

    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"

    assert scale > 0.0, "Argument scale should be positive"

    output_size = img.size
    center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
    kwargs = {"fillcolor": fillcolor} if PILLOW_VERSION[0] == '5' else {}
    return img.transform(output_size, Image.AFFINE, matrix, resample, **kwargs)


def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        PIL Image: Brightness adjusted image.
    """
    check_type(img)

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        PIL Image: Contrast adjusted image.
    """
    check_type(img)

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL Image: Saturation adjusted image.
    """
    check_type(img)

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


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
        img (PIL Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        PIL Image: Hue adjusted image.
    """
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    check_type(img)

    input_mode = img.mode
    assert img.mode not in {'L', '1', 'I', 'F'}, \
        "Input image mode should not be {'L', '1', 'I', 'F'}"

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img
