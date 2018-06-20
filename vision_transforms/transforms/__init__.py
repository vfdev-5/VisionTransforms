
from vision_transforms.transforms.base import BaseTransform
from vision_transforms.transforms.geometry import *


def _set_backend(mod):
    import vision_transforms.transforms.geometry
    vision_transforms.transforms.geometry.F = mod
