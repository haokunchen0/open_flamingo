"""
Cache CLIP text features for ImageNet labels.
"""
import argparse
import sys
import os
import torch
from tqdm import tqdm
from classification_utils import IMAGENET_CLASSNAMES,CUB_CLASSNAMES,STANFORD_CAR_CLASSNAMES,STANFORD_DOG_CLASSNAMES,IMAGENET_CLASSNAMES_SUB
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
    )
)
from eval.rices import RICES_Text

# 定义ImageNet的标签
labels_dict = {
    "label_imagenet_sub": IMAGENET_CLASSNAMES_SUB,
    "label_imagenet": IMAGENET_CLASSNAMES,
    "label_CUB200": CUB_CLASSNAMES,
    "list_STANFORD_CAR": STANFORD_CAR_CLASSNAMES,
    "list_STANFORD_DOG": STANFORD_DOG_CLASSNAMES,
    "list_test": ['fly', 'bee', 'ant']
}
parser = argparse.ArgumentParser()
parser.add_argument(
    "--label_name",
    type=str,
    required=True,
    help="Name of the label to choose.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory to save the cached text features.",
)
parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
parser.add_argument("--batch_size", default=256, type=int)

def main():
    args = parser.parse_args()
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    selected_list = labels_dict.get(args.label_name)
    print(selected_list)
    if selected_list:
        print("Caching {} labels...".format(selected_list))
    
    rices_text_dataset = RICES_Text(
        dataset=None,
        text_label=selected_list,
        device=device_id,
        batch_size=256,
        vision_encoder_path=args.vision_encoder_path,
        vision_encoder_pretrained=args.vision_encoder_pretrained,
)

    
    # 注意：这里我们保存文本特征而不是图像特征
    torch.save(
        rices_text_dataset.text_features,
        os.path.join(args.output_dir, args.label_name+".pkl"),
    )

if __name__ == "__main__":
    main()
