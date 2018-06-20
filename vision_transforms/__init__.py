
from vision_transforms.transforms import *


def set_image_backend(backend):
    """

    Args:
        backend (str): "opencv" or "pillow"

    Returns:a

    """
    assert backend in ("opencv", "pillow")

    from vision_transforms.transforms import _set_backend
    import vision_transforms.functional

    if backend == "opencv":
        from vision_transforms.functional import opencv as cv_image
        _set_backend(cv_image)
        vision_transforms.functional.image = cv_image
    else:
        from vision_transforms.functional import pillow as pil_image
        _set_backend(pil_image)
        vision_transforms.functional.image = pil_image


def get_image_backend():
    from vision_transforms.functional import image
    return image.backend_name

