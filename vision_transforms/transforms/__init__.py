
from vision_transforms.transforms.base import *
from vision_transforms.transforms.geometry import *
from vision_transforms.transforms.color import *


def _set_backend(mod):
    import vision_transforms.transforms.geometry
    import vision_transforms.transforms.color
    vision_transforms.transforms.geometry.F = mod
    vision_transforms.transforms.color.F = mod
