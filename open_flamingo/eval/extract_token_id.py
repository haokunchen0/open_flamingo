from transformers import AutoTokenizer
from classification_utils import IMAGENET_CLASSNAMES
# 设置文件夹路径
tokenizer_path = "/data/share/mpt-7b"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# 假设你的标签列表如下：
labels = IMAGENET_CLASSNAMES

# 找出长度为1的标签
single_token_labels = [label for label in labels if len(tokenizer.tokenize(label)) == 1]

print(single_token_labels)
