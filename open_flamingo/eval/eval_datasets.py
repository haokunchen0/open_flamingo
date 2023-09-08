from torchvision.datasets import ImageFolder

from open_flamingo.eval.classification_utils import IMAGENET_CLASSNAMES


class ImageNetDataset(ImageFolder):
    """Class to represent the ImageNet1k dataset."""

    def __init__(self, root, **kwargs):
        super().__init__(root=root, **kwargs)
        self.class_id_to_name = dict(
            zip(range(len(IMAGENET_CLASSNAMES)), IMAGENET_CLASSNAMES)
        )

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        target_label = self.class_id_to_name[target]
        return {
            "id": idx,
            "image": sample,
            "class_id": target,  # numeric ID of the ImageNet class
            "class_name": target_label,  # human-readable name of ImageNet class
        }