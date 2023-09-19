"""
Cache CLIP features for ImageNet images in training split in preparation for RICES
"""
import argparse
import sys
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
    )
)
from rices import RICES
# 使用我们之前定义的 ImageNetSubsetDataset
from eval_datasets import ImageNetSubsetDataset
from classification_utils import IMAGENET_CLASSNAMES_SUB
parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory to save the cached features.",
)
parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
parser.add_argument("--batch_size", default=256, type=int)

## Imagenet dataset
parser.add_argument("--imagenet_root", type=str, default="/tmp")

def main():
    args = parser.parse_args()
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    
    print("Caching ImageNet Subset...")
    # 使用 ImageNetSubsetDataset 以及子集类名
    train_dataset = ImageNetSubsetDataset(
        root=(os.path.join(args.imagenet_root, "train"))
    )
    rices_dataset = RICES(
        train_dataset,
        device_id,
        args.batch_size,
        vision_encoder_path=args.vision_encoder_path,
        vision_encoder_pretrained=args.vision_encoder_pretrained,
    )
    torch.save(
        rices_dataset.features,
        os.path.join(args.output_dir, "imagenet_sub_3.pkl"),
    )

if __name__ == "__main__":
    main()
