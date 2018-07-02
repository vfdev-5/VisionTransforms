# Another concepts for vision transforms for PyTorch/Torchvision

Idea is to provide basic brick to transform image/mask/bounding box/keypoints and user composes the transformation/augmentations according to the input datapoint.

Transforms can be made similar to models that inherit of `torch.nn.Module`.

Simple transformations are like `nn.Sequential`, more complicated should inherit from base transformation and 
overload `__call__` method.


## Transformation use-cases


### Simple example

Datapoint is `(img, label)`:

```python
transforms = vt.Sequential(
    vt.RandomAffine(degrees=(-45, 45), translate=(0.3, 0.3), scale=(0.75, 1.2), shear=(-15, 15), resample=PIL.Image.BILINEAR),
    vt.RandomCrop(size=224, padding=0),
)


def data_transform(datapoint):    
    return transforms(datapoint[0]), datapoint[1]
```

This case is almost the same as torchvision

### Another example

More complicated example, datapoint is `((img, scalars), (mask, bboxes, labels))` (e.g. Mask-RCNN)


```python

class DataTransform(vt.BaseTransform):
    
    def __init__(self):
        
        translate_scale_params = {
            'translate': (0.2, 0.2),
            'scale': (0.7, 1.3)
        }
        self.random_affine = vt.RandomAffine(degrees=0, **translate_scale_params, resample=PIL.Image.BICUBIC)
        self.mask_random_affine = vt.RandomAffine(degrees=0, **translate_scale_params, resample=PIL.Image.NEAREST)        
        self.bbox_random_affine = vt.BBoxRandomAffine(input_canvas_size=310, **translate_scale_params)
        
        self.random_crop = vt.RandomCrop(size=224)
        self.bbox_random_crop = vt.BBoxRandomCrop(input_canvas_size=310, size=224)
        
        self.img_geom = vt.Sequential(
            self.random_affine,
            self.random_crop,
        )
        self.mask_geom = vt.Sequential(
            self.mask_random_affine,
            self.random_crop,
        )        
        self.bbox_geom = vt.Sequential(
            self.bbox_random_affine,
            self.bbox_random_crop,
        )        
        self.img_color = vt.ColorJitter(hue=0.5, saturation=1.0)
        
    def __call__(self, datapoint, rng=None):
        
        x, y = datapoint
        img_rgb, scalars = x
        mask, bboxes, labels = y

        t_img_rgb = self.img_geom(img_rgb, rng)
        t_img_rgb = self.img_color(t_img_rgb)

        t_mask = self.mask_geom(mask, rng)        
        t_bboxes = self.bbox_geom(bboxes, rng)    
        
        return (t_img_rgb, scalars), (t_mask, t_bboxes, labels)
        

dtf = DataTransform()

        
def data_transform(datapoint):
    return dtf(datapoint)
```

## Random state controlling

All random parameters should use `random` built-in module.

All transformation `__call__` functions receive input and rng (can be None) as argument and setup random state before
transforming the data.


## Backends 

Image/Mask reading and transformations can be implemented using various backends: 
- Pillow
- Opencv/Numpy
- torch.tensor (optionally) **NOT IMPLEMENTED YET**

Bounding boxes/Keypoints are defined as `numpy.ndarray` and operations can be done using different backends:
- numpy
- torch.tensor (optionally) **NOT IMPLEMENTED YET**


## Concepts in practice

See the [example](https://github.com/vfdev-5/VisionTransforms/blob/master/examples/basic.ipynb)


