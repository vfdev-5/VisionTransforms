
from abc import ABCMeta, abstractmethod
import random
import numbers


__all__ = ["BaseTransform", "BaseTransformWithCanvas", "Sequential",
           "Lambda", "BBoxLambda", ]


class BaseTransform(object):
    """

    """
    __metaclass__ = ABCMeta

    def _setup_rng(self, rng=None):
        if rng is not None:
            random.setstate(rng)

    @abstractmethod
    def __call__(self, x, rng=None):
        pass

    def __repr__(self):
        params = ",\n\t".join(["{}={}".format(k, v) for k, v in self.__dict__.items()])
        out = self.__class__.__name__ + "(\n\t{})".format(params)
        return out


class BaseTransformWithCanvas(BaseTransform):

    __metaclass__ = ABCMeta

    def __init__(self, input_canvas_size, **kwargs):
        super(BaseTransformWithCanvas, self).__init__(**kwargs)
        if isinstance(input_canvas_size, numbers.Number):
            self.input_canvas_size = [int(input_canvas_size), int(input_canvas_size)]
        else:
            self.input_canvas_size = input_canvas_size

    def _check_input_canvas_size(self, input_canvas_size):
        if input_canvas_size is None and self.input_canvas_size is None:
            raise ValueError("Parameter input_canvas_size is not specified")

    def _get_input_canvas_size(self, input_canvas_size):
        input_canvas_size = self.input_canvas_size if input_canvas_size is None else input_canvas_size
        if isinstance(input_canvas_size, numbers.Number):
            return [int(input_canvas_size), int(input_canvas_size)]
        return list(input_canvas_size)


class Sequential(BaseTransform):

    def __init__(self, *args):
        super(Sequential, self).__init__()
        self._transforms = []
        for idx, t in enumerate(args):
            self._transforms.append(t)

    def __call__(self, x, rng=None, **kwargs):
        for t in self._transforms:
            x = t(x, rng, **kwargs)
        return x

    def __len__(self):
        return len(self._transforms)


class Lambda(BaseTransform):
    """Apply a user-defined transform on the image.

    Args:
        func (function): function to be used for transform. Function's input is `img` and `rng`
    """

    def __init__(self, func):
        assert callable(func)
        super(Lambda, self).__init__()
        self.func = func

    def __call__(self, img, rng=None):
        return self.func(img, rng)


class BBoxLambda(BaseTransformWithCanvas):
    """Apply a user-defined transform on the bounding boxes.

    Args:
        func (function): function to be used for transform. Function's input is `bboxes`, `input_canvas_size` and `rng`
    """

    def __init__(self, input_canvas_size, func):
        assert isinstance(func, callable)
        super(BBoxLambda, self).__init__(input_canvas_size)
        self.func = func

    def __call__(self, bboxes, rng=None, input_canvas_size=None):
        self._check_input_canvas_size(input_canvas_size)
        input_canvas_size = self._get_input_canvas_size(input_canvas_size)
        return self.func(bboxes, input_canvas_size, self.rng)
