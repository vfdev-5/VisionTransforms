# Vision transforms new concepts for PyTorch

Transforms can be made similar to models that inherit of `torch.nn.Module`.

Simple transformations are like `nn.Sequential`, more complicated should inherit from base transformation and 
overload `__call__` method.

## Simple example

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

## Another example

More complicated example, datapoint is `((img, scalars), (mask, bboxes, labels))` (e.g. Mask-RCNN)


```python

class DataTransform(vt.BaseTransform):
    
    def __init__(self):
        
        random_affine_model = {
            'degrees': (-45, 45), 
            'translate': (0.3, 0.3), 
            'scale': (0.75, 1.2), 
            'shear': (-15, 15)
        }
        
        self.img_random_affine = vt.RandomAffine(**random_affine_model, resample=PIL.Image.BILINEAR)
        self.mask_random_affine = vt.RandomAffine(**random_affine_model, resample=PIL.Image.NEAREST)
        self.bbox_random_affine = vt.BBoxRandomAffine(**random_affine_model)
        
        self.random_crop = vt.RandomCrop(size=224, padding=0)
        self.bbox_random_crop = vt.BBoxRandomCrop(size=224, canvas_size=(512, 512), padding=0)
        
        self.img_geom = vt.Sequential(
            self.img_random_affine,
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
        
        self.img_color = vt.ColorJitter(hue=0.2, saturation=0.2)
        
    def __call__(self, datapoint):
        
        rng = self.get_rng()
        
        x, y = datapoint
        img_rgb, scalars = x
        mask, bboxes, labels = y
                
        t_img_rgb = self.img_geom(img_rgb, rng)
        t_img_rgb = self.img_color(t_img_rgb, rng)
                       
        t_mask = self.mask_geom(mask, rng)        
        t_bboxes = self.bbox_geom(bboxes, rng)    
    
        return ((t_img_rgb, scalars), (t_mask, t_bboxes, labels))


dtf = DataTransform()

        
def data_transform(datapoint):
    return dtf(datapoint)
```

