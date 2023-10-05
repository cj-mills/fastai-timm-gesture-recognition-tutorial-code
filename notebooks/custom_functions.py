import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2  as transforms
from fastcore.transform import Transform

class ConditionalTypeTransform:
    def __init__(self, input_type, transforms_list):
        self.input_type = input_type
        self.transforms = transforms.Compose(transforms_list)

    def __call__(self, x):
        if isinstance(x, self.input_type):
            return self.transforms(x)
        else:
            return x
