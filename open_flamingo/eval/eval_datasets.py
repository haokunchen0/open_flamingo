from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from classification_utils import IMAGENET_CLASSNAMES, IMAGENET_CLASSNAMES_SUB
from classification_utils import STANFORD_CAR_ID_TO_LABEL, CUB_CLASSNAMES, STANFORD_DOG_CLASSNAMES

import os
import scipy.io
from PIL import Image

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
        
class ImageNetSubsetDataset(ImageFolder):
    def __init__(self, root, **kwargs):
        super().__init__(root=root, **kwargs)
        
        # 将IMAGENET_CLASSNAMES映射到self.classes
        self.full_class_to_foldername = dict(zip(IMAGENET_CLASSNAMES, self.classes))
        
        # 为子集创建一个新的映射
        self.subset_class_to_foldername = {cls: self.full_class_to_foldername[cls] for cls in ['fly', 'bee', 'ant']}
        subset_folder_names = list(self.subset_class_to_foldername.values())
        
        # 只保留子集类别的数据
        self.samples = [s for s in self.samples if self.classes[s[1]] in subset_folder_names]
        self.imgs = self.samples
        
        # 建立新的类名到索引的映射
        self.class_name_to_id = {name: idx for idx, name in enumerate(['fly', 'bee', 'ant'])}
        
    def __getitem__(self, idx):
        sample, old_target = super().__getitem__(idx)
        
        # 获取正确的子集类名和对应的索引
        class_name = self.classes[old_target]
        for name, id in self.class_name_to_id.items():
            if self.full_class_to_foldername[name] == class_name:
                target = id
                break
        
        return {
            "id": idx,
            "image": sample,
            "class_id": target,  # numeric ID of the ImageNet subset class
            "class_name": name,  # human-readable name of ImageNet class
        }

class CUB200Dataset(Dataset):
    """Class to represent the CUB dataset."""

    def __init__(self, root, transform=None, train=True):
        self.root = root
        self.transform = transform
        self.train = train
        self.image_paths = []
        self.labels = []
        self.class_id_to_name = dict(zip(range(len(CUB_CLASSNAMES)), CUB_CLASSNAMES))

        image_names = self._get_image_names()

        # Read image_class_labels.txt
        label_file = os.path.join(root, 'image_class_labels.txt')
        with open(label_file, 'r') as f:
            for line in f:
                image_id, class_id = line.strip().split()
                self.image_paths.append(os.path.join(root, 'images', image_names[int(image_id) - 1]))
                self.labels.append(int(class_id) - 1)  # Class IDs are 1-indexed

        # Read train_test_split.txt
        split_file = os.path.join(root, 'train_test_split.txt')
        with open(split_file, 'r') as f:
            split_lines = f.readlines()

        # Filter images based on train or test split
        if self.train:
            self.image_paths = [path for i, path in enumerate(self.image_paths) if int(split_lines[i].strip().split()[1]) == 1]
            self.labels = [label for i, label in enumerate(self.labels) if int(split_lines[i].strip().split()[1]) == 1]
        else:
            self.image_paths = [path for i, path in enumerate(self.image_paths) if int(split_lines[i].strip().split()[1]) == 0]
            self.labels = [label for i, label in enumerate(self.labels) if int(split_lines[i].strip().split()[1]) == 0]

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        class_name = CUB_CLASSNAMES[label]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return {
            "id": idx,
            "image": img,
            "class_id": label,  # numeric ID of the CUB-200-201 class
            "class_name": class_name  # Modify this if you want to include class names
        }
    
    def _get_image_names(self):
        image_names = []
        with open(os.path.join(self.root, 'images.txt'), 'r') as f:
            for line in f:
                image_id, image_name = line.strip().split()
                image_names.append(image_name)
        return image_names
    
    def __len__(self) -> int:
        return len(self.image_paths)

class StanfordCarDataset(ImageFolder):
    """Class to represent the Stanford Car dataset."""

    def __init__(self, root, **kwargs):
        super().__init__(root=root, **kwargs)
        self.class_id_to_name = STANFORD_CAR_ID_TO_LABEL

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        target_label = self.class_id_to_name[target]
        return {
            "id": idx,
            "image": sample,
            "class_id": target,  # numeric ID of the ImageNet class
            "class_name": target_label,  # human-readable name of ImageNet class
        }
    
class StanfordDogDataset(Dataset):
    """Class to represent the Stanford Dog dataset."""

    def __init__(self, root, transform=None, train=True):
        self.root = root
        self.transform = transform
        self.train = train
        self.image_paths = []
        self.labels_ = []
        self.labels = []
        self.split_file = "train_list.mat" if self.train else "test_list.mat"
        self.class_id_to_name = dict(zip(range(len(STANFORD_DOG_CLASSNAMES)), STANFORD_DOG_CLASSNAMES))

        self.class_name2id = dict(
            zip(STANFORD_DOG_CLASSNAMES, range(len(STANFORD_DOG_CLASSNAMES)))
        )
        # Load split_file
        file_list = scipy.io.loadmat(os.path.join(self.root, self.split_file))['file_list']
        for item in file_list:
            file_path = item[0][0]
            self.image_paths.append(os.path.join(root, "images", file_path))
            class_name = file_path.split("/")[0][10:]
            self.labels_.append(class_name)
            self.labels.append(self.class_name2id[class_name])
            


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path, target_label = self.image_paths[idx], self.labels_[idx]
        sample = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        class_id = self.class_name2id[target_label]
        return {
            "id": idx,
            "image": sample,
            "class_id": class_id,  # numeric ID of the ImageNet class
            "class_name": target_label,  # human-readable name of ImageNet class
        }
        