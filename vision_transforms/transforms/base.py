
from abc import ABCMeta, abstractmethod
import random


class BaseTransform(object):
    """

    """
    __metaclass__ = ABCMeta

    def _setup_rng(self, rng=None):
        if rng is not None:
            random.setstate(rng)

    def get_rng(self):
        """Method to get random state

        Returns:
            RNG state
        """
        return random.getstate()

    @abstractmethod
    def __call__(self, x, rng=None):
        pass


class Sequential(BaseTransform):

    def __init__(self, *args):
        super(Sequential, self).__init__()
        self._transforms = []
        for idx, t in enumerate(args):
            self._transforms.append(t)

    def __call__(self, x, rng=None):
        pass

    def __len__(self):
        return len(self._transforms)
