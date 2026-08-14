"""
Evaluate a trained model on clean test set and CIFAR-100-C corruptions.

Usage:
    python -m scripts.evaluate --checkpoint outputs/students/asd_.../epoch_0240_best.pth \
                               --arch resnet18 --dataset cifar100
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.models import build_model
from utils.helpers import load_checkpoint
from data.datasets import get_dataloaders
from core.evaluator import full_evaluation, print_corruption_table


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--corruption_root", type=str, default="./data/CIFAR-100-C")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    num_classes = {"cifar100": 100, "cifar10": 10, "imagenet100": 100, "tinyimagenet": 200}[args.dataset]
    model = build_model(args.arch, num_classes, args.dataset)
    load_checkpoint(args.checkpoint, model, device=args.device)
    model = model.to(args.device)

    _, test_loader = get_dataloaders(args.dataset, args.data_root, batch_size=128)

    output_path = args.output or args.checkpoint.replace(".pth", "_eval.json")

    print(f"Evaluating {args.arch} from {args.checkpoint}")
    results = full_evaluation(
        model=model,
        test_loader=test_loader,
        corruption_data_root=args.corruption_root,
        device=args.device,
        output_path=output_path,
        dataset=args.dataset,
    )

    print(f"\nClean Accuracy: {results['clean']['clean_acc']:.2f}%")
    print(f"Clean Error:    {results['clean']['clean_err']:.2f}%")
    print(f"ECE:            {results['clean'].get('ece', float('nan')):.4f}")

    if results["corruption"]["per_corruption"]:
        print_corruption_table(results["corruption"])

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
