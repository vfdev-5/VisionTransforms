import numbers
import collections

from PIL import Image, ImageOps


backend_name = "pillow"


def check_type(img):
    if not isinstance(img, Image.Image):
        raise TypeError("Input image should be PIL.Image.Image")


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


def pad(img, padding, fill=0):
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

    return ImageOps.expand(img, border=padding, fill=fill)
