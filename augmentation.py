import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class SimCLRTransform(object):
    def __init__(self, resize=224, s=1.0):
        """
        A default setting for augmentations in a SimCLR.

        - Args
            s (float): the strength of color distortion.
        """
        blur_kernel = int(resize*0.1) if int(resize*0.1) % 2 else int(resize*0.1)+1
        self.transforms = A.Compose(
            [
                A.RandomResizedCrop(height=resize, width=resize, scale =[0.08, 1.0], ratio=[3/4, 4/3]),
                A.Flip(),
                self.get_color_distortion(s),
                A.GaussianBlur(blur_limit = (blur_kernel, blur_kernel), sigma_limit=(0.1, 2.0)),
                A.Normalize(),
                ToTensorV2()
            ]
        )
    
    def get_color_distortion(self,s):
        rand_color_jitter = A.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s, p = 0.8)
        rand_gray = A.ToGray(p = 0.2)

        color_distortion = A.Compose(
            [
                rand_color_jitter,
                rand_gray
            ]
        )

        return color_distortion
    
    def __call__(self, img):
        return self.transforms(image=img)

class BaseTransform(object):
    def __init__(self, crop=224):
        self.transforms = A.Compose(
            [
                A.RandomResizedCrop(crop, crop),
                A.HorizontalFlip(),
                A.Normalize(),
                ToTensorV2()
            ]
        )
    
    def __call__(self, img):
        return self.transforms(image=img)