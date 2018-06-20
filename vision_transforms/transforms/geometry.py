import random
import numbers

from vision_transforms.transforms import BaseTransform
from vision_transforms.functional import image as F
from vision_transforms.functional import bbox as B
from vision_transforms.functional import keypoints as K


__all__ = ["RandomCrop", "BBoxRandomCrop", ]


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
            img = F.pad(img, padding=self.padding)

        ih, iw, _ = F.shape(img)
        x, y, w, h = self.get_params((ih, iw), self.size)
        return F.crop(img, x, y, w, h)


class BBoxRandomCrop(RandomCrop):
    """Random crop transformation for bounding box

    """

    def __init__(self, input_canvas_size, output_canvas_size, padding=0):
        """

        Args:
            input_canvas_size (int or sequence): input canvas size (h, w), e.g. corresponding image size
            output_canvas_size (int or sequence): output canvas size (h, w)
            padding (int or sequence, optional): Optional padding on each border
                of the image. Default is 0, i.e no padding. If a sequence of length
                4 is provided, it is used to pad left, top, right, bottom borders
                respectively.

        """
        super(BBoxRandomCrop, self).__init__(output_canvas_size, padding=padding)
        if isinstance(input_canvas_size, numbers.Number):
            self.input_canvas_size = (int(input_canvas_size), int(input_canvas_size))
        else:
            self.input_canvas_size = input_canvas_size

    def __call__(self, bboxes, rng=None):

        if self.padding > 0:
            bboxes = B.pad(bboxes, padding=self.padding)

        self._setup_rng(rng)
        x, y, w, h = self.get_params(self.input_canvas_size, self.size)
        return B.crop(bboxes, x, y, w, h)
