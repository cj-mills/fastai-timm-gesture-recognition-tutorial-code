import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2  as transforms
from fastcore.transform import Transform
from fastai.vision.core import PILImage

class TorchvisionComposeWrapper(Transform):
    """
    Wrap Torchvision transforms for use within Fastai's pipeline.

    Args:
    - transforms_list (list): List of Torchvision transforms.
    """
    
    def __init__(self, transforms_list):
        """Initialize with a list of torchvision transforms."""
        self.transforms = transforms.Compose(transforms_list)

    def encodes(self, x: PILImage) -> PILImage:
        """Apply the Torchvision transformations to the PILImage."""
        return PILImage(self.transforms(x))
