
from vision_transforms import get_image_backend, set_image_backend


def test_set_image_backend():
    set_image_backend("pillow")
    from vision_transforms.functional import image as F
    assert F.__package__ == "vision_transforms.functional.pillow"
    set_image_backend("opencv")
    from vision_transforms.functional import image as F
    assert F.__package__ == "vision_transforms.functional.opencv"


def test_get_image_backend():
    set_image_backend("pillow")
    assert "pillow" == get_image_backend()
    set_image_backend("opencv")
    assert "opencv" == get_image_backend()
