import random
import numbers

from vision_transforms.transforms import BaseTransform
from vision_transforms.transforms.base import BaseTransformWithCanvas
from vision_transforms.functional import image as F
from vision_transforms.functional import bbox as B
# from vision_transforms.functional import keypoints as K


_all_image_funcs = ["RandomCrop", "RandomAffine", ]
_all_bbox_funcs = ["BBox{}".format(f) for f in _all_image_funcs]
_all_kp_funcs = []

__all__ = _all_image_funcs + _all_bbox_funcs + _all_kp_funcs


class RandomCrop(BaseTransform):
    """Random crop transformation for image

    """

    def __init__(self, size, padding=0):
        """

        Args:
            size (int or sequence): output size (h, w)
            padding (int or sequence, optional): Optional padding on each border
                of the image. Default is 0, i.e no padding. If a sequence of length
                4 is provided, it is used to pad left, top, right, bottom borders
                respectively.
        """
        super(RandomCrop, self).__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(input_size, output_size):
        """Get parameters for a random crop.

        Args:
            input_size (tuple): Input image size (h, w)
            output_size (tuple): Expected output size of the crop (h, w)

        Returns:
            tuple: params (i, j, h, w) to be passed for a random crop.
        """
        h, w = input_size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        y = random.randint(0, h - th)
        x = random.randint(0, w - tw)
        return x, y, th, tw

    def __call__(self, img, rng=None):
        self._setup_rng(rng)

        if self.padding > 0:
            img = F.pad(img, padding=self.padding, mode='constant')

        ih, iw, _ = F.shape(img)
        x, y, w, h = self.get_params((ih, iw), self.size)
        return F.crop(img, x, y, w, h)


class BBoxRandomCrop(BaseTransformWithCanvas, RandomCrop):
    """Random crop transformation for bounding box

    """

    def __init__(self, input_canvas_size, size, padding=None):
        """

        Args:
            input_canvas_size (int or sequence): input canvas size (h, w), e.g. corresponding image size
            size (int or sequence): output canvas size (h, w)
            padding (int or sequence, optional): Optional padding on each border
                of the image. Default is 0, i.e no padding. If a sequence of length
                4 is provided, it is used to pad left, top, right, bottom borders
                respectively.

        """
        super(BBoxRandomCrop, self).__init__(input_canvas_size=input_canvas_size, size=size, padding=padding)

        if self.padding is not None:
            pad_left, pad_top, pad_right, pad_bottom = B._get_ltrb_padding(self.padding)
            self.input_canvas_size[1] += (pad_left + pad_right)
            self.input_canvas_size[0] += (pad_top + pad_bottom)

    def __call__(self, bboxes, rng=None):

        if self.padding is not None:
            bboxes = B.pad(bboxes, padding=self.padding, inplace=False)

        self._setup_rng(rng)
        x, y, w, h = self.get_params(self.input_canvas_size, self.size)
        return B.crop(bboxes, x, y, w, h)


class RandomAffine(BaseTransform):
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to desactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Will not apply shear by default
        resample (int, optional): This parameter depends on backend
        fillcolor (int): Optional fill color for the area outside the transform in the output image.
            If backend is `pillow`, this option is enabled for Pillow>=5.0.0
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=0, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (round(random.uniform(-max_dx, max_dx)),
                            round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img, rng=None):
        self._setup_rng(rng)
        ih, iw, _ = F.shape(img)
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, (ih, iw))
        return F.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)


class BBoxRandomAffine(BaseTransformWithCanvas, RandomAffine):
    """Random affine transformation of the bounding box keeping canvas center invariant

    Args:
        input_canvas_size (int or sequence): input canvas size (h, w), e.g. corresponding image size
        degrees (sequence or float or int): Unused as does not make sense. Remains for compatibility with other affine
            transformations
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Unused as does not make sense. Remains for compatibility with
            other affine transformations
    """

    def __init__(self, input_canvas_size, degrees=0, translate=None, scale=None, shear=None):
        super(BBoxRandomAffine, self).__init__(input_canvas_size=input_canvas_size,
                                               degrees=0, translate=translate, scale=scale, shear=None)

    def __call__(self, bboxes, rng=None):
        self._setup_rng(rng)
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.input_canvas_size)
        center = (self.input_canvas_size[0] * 0.5 + 0.5, self.input_canvas_size[1] * 0.5 + 0.5)
        return B.affine(bboxes, center, *ret)

