import argparse
import importlib
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import utils
import math

from eval_datasets import ImageNetDataset,CUB200Dataset,StanfordCarDataset, StanfordDogDataset,ImageNetSubsetDataset

from rices import RICES,RICES_Text
from tqdm import tqdm

from classification_utils import IMAGENET_CLASSNAMES_SUB,IMAGENET_CLASSNAMES, CUB_CLASSNAMES, STANFORD_CAR_CLASSNAMES, STANFORD_DOG_CLASSNAMES
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    help="Model name. Currently only `OpenFlamingo` is supported.",
    default="open_flamingo",
)
parser.add_argument(
    "--results_file", type=str, default=None, help="JSON file to save results"
)
# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0, 4, 8, 16, 32], type=int)
parser.add_argument(
    "--num_trials",
    type=int,
    default=1,
    help="Number of trials to run for each shot using different demonstrations",
)
parser.add_argument(
    "--trial_seeds",
    nargs="+",
    type=int,
    default=[42],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=-1,
    help="Number of samples to evaluate on. -1 for all samples.",
)
parser.add_argument(
    "--query_set_size", type=int, default=2048, help="Size of demonstration query set"
)

#parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--batch_size_map", type=str, default="0:180,4:1,8:32")

parser.add_argument(
    "--no_caching_for_classification",
    action="store_true",
    help="Whether to skip using key-value caching for classification evals, which usually speeds it up.",
)
parser.add_argument(
    "--classification_prompt_ensembling",
    action="store_true",
    help="Whether to use prompt ensembling (average log-likelihoods over permutations of in-context examples)",
)
parser.add_argument(
    "--rices",
    action="store_true",
    help="Whether to use RICES for evaluation. If False, uses random demonstrations.",
)
parser.add_argument(
    "--rices_text",
    action="store_true",
    help="Whether to use RICES_text for evaluation. If False, uses random demonstrations.",
)

parser.add_argument(
    "--rices_vision_encoder_path",
    default="ViT-L-14",
    type=str,
    help="CLIP vision encoder to use for RICES if cached_demonstration_features is None.",
)
parser.add_argument(
    "--rices_vision_encoder_pretrained",
    default="openai",
    type=str,
    help="CLIP vision encoder to use for RICES if cached_demonstration_features is None.",
)
parser.add_argument(
    "--cached_demonstration_features",
    default=None,
    help="Directory where rices features for all choices of in-context examples are stored as a pkl file with the dataset name. If None, features are re-computed by script.",
)
parser.add_argument(
    "--label_cached_demonstration_features",
    default=None,
    help="for ML image to label..."
)
# Dataset arguments
## load dataset
parser.add_argument("--dataset_root", type=str, default="/tmp")

# Multilabel
parser.add_argument(
    "--multilabel",
    action="store_true",
    help="Whether to use multi-label for evaluation."
)
# Distributed evaluation
parser.add_argument(
    "--dist-url",
    default="env://",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--horovod",
    default=False,
    action="store_true",
    help="Use horovod for distributed training.",
)
parser.add_argument(
    "--no-set-device-rank",
    default=False,
    action="store_true",
    help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
)
parser.add_argument(
    "--pkl_name",
    type=str,
    required=True,
    help="Name of the label.pkl to choose.",
)
# Choose dataset to evaluate
parser.add_argument("--dataset_name", type=str, default="imagenet")
# dynamic adjusting batchsize mapping
def parse_batch_size_map(batch_size_map_str):
    mapping = {}
    pairs = batch_size_map_str.split(',')
    for pair in pairs:
        shot, batch_size = pair.split(':')
        mapping[int(shot)] = int(batch_size)
    return mapping


def main():
    args, leftovers = parser.parse_known_args()
    module = importlib.import_module(f"open_flamingo.eval.models.{args.model}")
    args.batch_size_map = parse_batch_size_map(args.batch_size_map)
    
    model_args = {
        leftovers[i].lstrip("-"): leftovers[i + 1] for i in range(0, len(leftovers), 2)
    }
    eval_model = module.EvalModel(model_args)

    # set up distributed evaluation
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    eval_model.set_device(device_id)
    eval_model.init_distributed()

    if args.model != "open_flamingo" and args.shots != [0]:
        raise ValueError("Only 0 shot eval is supported for non-open_flamingo models")

    if len(args.trial_seeds) != args.num_trials:
        raise ValueError("Number of trial seeds must be == number of trials.")

    results = defaultdict(list)
    
    print(f"Evaluating on {args.dataset_name} Dataset...")

    # load cached demonstration features for RICES
    if args.cached_demonstration_features is not None and args.rices:
        cached_features = torch.load(
            f"{args.cached_demonstration_features}/{args.pkl_name}", map_location="cpu"
        )
    elif args.cached_demonstration_features is not None and args.rices_text:
        cached_features = torch.load(
            f"{args.cached_demonstration_features}/{args.pkl_name}", map_location="cpu"
        )
    else:
        cached_features = None
    if args.multilabel:
        label_cached_features = torch.load(
            f"{args.label_cached_demonstration_features}", map_location="cpu"
        )
    else:
        label_cached_features = None
    for shot in args.shots:
        batch_size = args.batch_size_map.get(shot, args.batch_size_map[0])  # 使用默认值，如果没有为该 shot 定义 batch_size的话
        print(f"Now evaluating on {batch_size} samples...")
        scores = []
        for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
            dataset_score = evaluate_classification(
                args,
                eval_model=eval_model,
                num_shots=shot,
                batch_size=batch_size,
                seed=seed,
                no_kv_caching=args.no_caching_for_classification,
                dataset_name=args.dataset_name,
                cached_features=cached_features,
                label_cached_features=label_cached_features,
                use_prompt_ensembling=args.classification_prompt_ensembling,
            )
            if args.rank == 0:
                print(
                    f"Shots {shot} Trial {trial} "
                    f"{args.dataset_name} score: {dataset_score}"
                )
                scores.append(dataset_score)

        if args.rank == 0:
            print(f"Shots {shot} Mean {args.dataset_name} score: {np.nanmean(scores)}")
            results[f"{args.dataset_name}"].append(
                {
                    "shots": shot,
                    "trials": scores,
                    "mean": np.nanmean(scores),
                    "stddev": np.nanstd(scores),
                }
            )
                
    if args.rank == 0 and args.results_file is not None:
        with open(args.results_file, "a") as f:
            json.dump(results, f)


def evaluate_classification(
    args: argparse.Namespace,
    eval_model,
    batch_size: int,
    seed: int = 42,
    num_shots: int = 8,
    dataset_name: str = "imagenet",
    cached_features=None,
    label_cached_features=None,
    no_kv_caching=False,
    use_prompt_ensembling: bool = False,
):
    """
    Evaluate a model on classification dataset.

    Args:
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): random seed. Defaults to 42.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        no_kv_caching (bool): whether to disable key-value caching
        dataset_name (str, optional): dataset name. Defaults to "imagenet".
        cached_features (tensor, optional): cached demonstration features for RICES. Defaults to None.

    Returns:
        float: accuracy score
    """
    if args.model != "open_flamingo":
        raise NotImplementedError(
            "evaluate_classification is currently only supported for OpenFlamingo"
        )
    # get_prompt_for_labels_lambda = lambda labels: eval_model.get_imagenet_prompt(label=",".join(labels))
    if dataset_name == "cub200":
        train_dataset = CUB200Dataset(
            root=args.dataset_root
        )
        test_dataset = CUB200Dataset(
            root=args.dataset_root,
            train=False
        )
        prompt_fn = lambda x: eval_model.get_imagenet_prompt(label=x["class_name"])
        all_class_names = CUB_CLASSNAMES
        k = 5
    elif dataset_name == "imagenet":
        train_dataset = ImageNetDataset(
            root=(os.path.join(args.dataset_root, "train"))
        )
        test_dataset = ImageNetDataset(
            root=(os.path.join(args.dataset_root, "test"))
        )
        prompt_fn = lambda x: eval_model.get_imagenet_prompt(label=x["class_name"])
        all_class_names = IMAGENET_CLASSNAMES
        k = 5
    elif dataset_name == 'imagenet_sub':
        train_dataset = ImageNetSubsetDataset(
            root=(os.path.join(args.dataset_root, "train")),
        )
        test_dataset = ImageNetSubsetDataset(
            root=(os.path.join(args.dataset_root, "test"))
        )
        prompt_fn = lambda x: eval_model.get_imagenet_prompt(label=x["class_name"])
        all_class_names = ['fly', 'bee', 'ant']
        k = 2
        # /data/hyh/car_data/car_data/
    elif dataset_name == "stanford_car":
        train_dataset = StanfordCarDataset(
            root=(os.path.join(args.dataset_root, "train"))
        )
        test_dataset = StanfordCarDataset(
            root=(os.path.join(args.dataset_root, "test"))
        )
        prompt_fn = lambda x: eval_model.get_imagenet_prompt(label=x["class_name"])
        all_class_names = STANFORD_CAR_CLASSNAMES
        k = 5
            
    elif dataset_name == "stanford_dog":
        train_dataset = StanfordDogDataset(
            root=args.dataset_root
        )
        test_dataset = StanfordDogDataset(
            root=args.dataset_root,
            train=False
        )
        prompt_fn = lambda x: eval_model.get_imagenet_prompt(label=x["class_name"])
        all_class_names = STANFORD_DOG_CLASSNAMES
        k = 5
    else:
        raise ValueError(f"Unsupported dataset {args.dataset_name}")

    labels = all_class_names
    class_id_to_name = dict(zip(range(len(all_class_names)), all_class_names))

    effective_num_shots = utils.compute_effective_num_shots(num_shots, args.model)

    np.random.seed(seed)
    test_dataloader = utils.prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        batch_size,
    )
    if args.rices:
        print("rices has been activated...")
        rices_dataset = RICES(
            train_dataset,
            eval_model.device,
            batch_size,
            cached_features=cached_features,
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
        )
    elif args.rices_text:
        print("rices_text has been activated...")
        rices_text_labels = RICES_Text(
            train_dataset,
            labels,
            eval_model.device,
            batch_size,
            cached_features=cached_features,
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
        )
    else:
        # subset of the training set to sample context images from
        query_set = utils.get_query_set(train_dataset, args.query_set_size)
    if args.multilabel:
        print("multilabel has been activated...")
        rices_text_forML = RICES_Text(
                        train_dataset,
                        labels,
                        eval_model.device,
                        batch_size,
                        cached_features=label_cached_features,
                        vision_encoder_path=args.rices_vision_encoder_path,
                        vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
                    )
    utils.random_seed(seed, args.rank)
    predictions = []
    for batch_idx, batch in tqdm(
        enumerate(test_dataloader),
        desc=f"Running inference {dataset_name}",
        disable=args.rank != 0,
    ):
        if args.rices:
            batch_demo_samples = rices_dataset.find(batch["image"], effective_num_shots)
        elif args.rices_text:
            batch_demo_labels = rices_text_labels.find(batch["image"], 1)
            #Currently rices_text's similiar label is set to be 1, wait and see...
            batch_demo_samples = rices_text_labels.get_images_from_labels(batch_demo_labels, effective_num_shots)
        else:
            batch_demo_samples = utils.sample_batch_demos_from_query_set(
                query_set, effective_num_shots, len(batch["image"])
            )
        # set up prompt ensembling
        num_permutations = (
            min(6, math.factorial(effective_num_shots)) if use_prompt_ensembling else 1
        )
        logprobs = []
        for _ in range(num_permutations):
            batch_images, batch_text = [], []
            for i in range(len(batch["image"])):
                if use_prompt_ensembling:
                    random.shuffle(batch_demo_samples[i])

                if effective_num_shots > 0:
                    context_images = [x["image"] for x in batch_demo_samples[i]]
                else:
                    context_images = []
                batch_images.append(context_images + [batch["image"][i]])
                if not args.multilabel:
                    context_text = "".join([prompt_fn(x) for x in batch_demo_samples[i]])
                    if num_shots == 0:
                        context_text = context_text.replace("<image>", "")
                        # Keep the text but remove the image tags for the zero-shot case
                    batch_text.append(
                        context_text
                        + prompt_fn({"class_name": None})
                    )
                else:
                    # 获取当前batch中所有图片的标签
                    shots_labels_for_current_batch = rices_text_forML.find([x['image'] for x in batch_demo_samples[i]], 2)

                    # 将每张图片的三个标签连接起来
                    labels_for_each_image = [",".join(shots_labels_for_current_batch[img_idx]) for img_idx in range(len(batch_demo_samples[i]))]

                    # 使用prompt_fn函数生成每张图片的prompt
                    prompts_for_each_image = [prompt_fn({"class_name": labels}) for labels in labels_for_each_image]

                    # 将所有图片的prompt连接起来
                    context_text = "".join(prompts_for_each_image)

                    batch_text.append(
                        context_text
                        + prompt_fn({"class_name": None})
                        )

        
                # batch_text: <image>Output:Dog,Cat,Lion<|endofchunk|><image>Output:class_name
            # get predicted class names
            logprobs.append(
                eval_model.get_rank_classifications(
                    args.multilabel,
                    batch_text,
                    batch_images,
                    all_class_names,
                    use_cache=(not no_kv_caching)
                )
            )

        # ensemble logprobs together
        logprobs = torch.mean(torch.stack(logprobs, dim=-1), dim=-1)

        predicted_classnames, predicted_logprobs = utils.get_predicted_classnames(
            logprobs,
            k,
            class_id_to_name,
        )

        # compute accuracy
        for i, topk in enumerate(predicted_classnames):
            y_i = batch["class_name"][i]
            score = torch.exp(
                predicted_logprobs[i][0] - torch.logsumexp(logprobs[i], dim=0)
            ).item()
            predictions.append(
                {
                    "id": batch["id"][i],
                    "gt_label": y_i,
                    "pred_label": topk[0],
                    "pred_score": score,
                }
            )

    # all gather
    all_predictions = [None for _ in range(args.world_size)]
    torch.distributed.all_gather_object(all_predictions, predictions)  # list of lists
    if args.rank != 0:
        return

    all_predictions = [
        item for sublist in all_predictions for item in sublist
    ]  # flatten


    # return top-1 accuracy
    acc1 = sum(
        int(pred["gt_label"] == pred["pred_label"]) for pred in all_predictions
    )
    return float(acc1) / len(all_predictions)

if __name__ == "__main__":
    main()
