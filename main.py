from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import AuthenticityPipeline
from utils.dataset import ensure_mock_dataset
from utils.image_io import create_demo_image
from utils.training import train_all, train_meta_classifier, train_pixel_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-modal real-vs-AI image authenticity verification system."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="Run end-to-end inference on one image.")
    predict.add_argument("--image", required=True, help="Path to an image.")
    predict.add_argument("--model-dir", default="models")
    predict.add_argument("--use-clip", action="store_true", help="Enable CLIP backend if installed.")
    predict.add_argument("--no-torch", action="store_true", help="Force lightweight pixel backends.")
    predict.add_argument("--compact", action="store_true", help="Only print label and confidence.")

    train = subparsers.add_parser("train", help="Train pixel streams and meta-classifier.")
    add_training_args(train)

    train_pixel = subparsers.add_parser("train-pixel", help="Train one or more pixel models.")
    add_training_args(train_pixel)
    train_pixel.add_argument(
        "--model",
        choices=["efficientnet", "alexnet", "googlenet", "all"],
        default="all",
    )

    train_meta = subparsers.add_parser("train-meta", help="Train the XGBoost meta-classifier.")
    add_training_args(train_meta)

    demo = subparsers.add_parser("demo", help="Create demo data, train fallbacks, and predict.")
    demo.add_argument("--data-dir", default="data")
    demo.add_argument("--model-dir", default="models")
    demo.add_argument("--use-clip", action="store_true")
    demo.add_argument("--no-torch", action="store_true")
    return parser


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default="data", help="Dataset root.")
    parser.add_argument("--model-dir", default="models", help="Model artifact directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--mock-if-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create tiny mock data/features when no dataset is present.",
    )
    parser.add_argument("--use-clip", action="store_true")
    parser.add_argument("--no-torch", action="store_true")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "predict":
        pipeline = AuthenticityPipeline(
            model_dir=args.model_dir,
            prefer_torch=not args.no_torch,
            use_clip=args.use_clip,
        )
        result = pipeline.predict(args.image)
        if args.compact:
            print(json.dumps({"label": result.label, "confidence": round(result.confidence, 2)}, indent=2))
        else:
            print(json.dumps(result.to_dict(include_details=True), indent=2))
        return

    if args.command == "train":
        result = train_all(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            mock_if_missing=args.mock_if_missing,
            limit=args.limit,
            prefer_torch=not args.no_torch,
            use_clip=args.use_clip,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "train-pixel":
        model_names = None if args.model == "all" else [args.model]
        result = train_pixel_models(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            model_names=model_names,
            mock_if_missing=args.mock_if_missing,
            limit=args.limit,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            prefer_torch=not args.no_torch,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "train-meta":
        result = train_meta_classifier(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            mock_if_missing=args.mock_if_missing,
            limit=args.limit,
            prefer_torch=not args.no_torch,
            use_clip=args.use_clip,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "demo":
        data_dir = Path(args.data_dir)
        real_path = create_demo_image(data_dir / "demo_single" / "real" / "demo_real.png", "real")
        create_demo_image(data_dir / "demo_single" / "fake" / "demo_fake.png", "fake")
        training_dir = ensure_mock_dataset(data_dir / "demo_training", per_class=4)
        train_result = train_all(
            data_dir=training_dir,
            model_dir=args.model_dir,
            mock_if_missing=True,
            limit=12,
            prefer_torch=not args.no_torch,
            use_clip=args.use_clip,
        )
        pipeline = AuthenticityPipeline(
            model_dir=args.model_dir,
            prefer_torch=not args.no_torch,
            use_clip=args.use_clip,
        )
        prediction = pipeline.predict(real_path)
        print(json.dumps({"training": train_result, "prediction": prediction.to_dict()}, indent=2))
        return


if __name__ == "__main__":
    main()
