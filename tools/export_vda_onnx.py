"""Export Video Depth Anything checkpoints to ONNX for the browser runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a Video-Depth-Anything checkpoint to ONNX with the same "
            "32-frame sliding window that the WebAssembly runner expects."
        )
    )
    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
        help="Path to the cloned Video-Depth-Anything repository",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the .pth checkpoint inside the Video-Depth-Anything repo",
    )
    parser.add_argument(
        "--encoder",
        choices=["vits", "vitb", "vitl"],
        default="vitl",
        help="Encoder size to instantiate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination ONNX file",
    )
    parser.add_argument(
        "--metric",
        action="store_true",
        help="Export the metric checkpoint variant",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Square input size (multiple of 14) used for the dummy export tensor",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    return parser.parse_args()


def load_model(repo: Path, encoder: str, metric: bool) -> torch.nn.Module:
    sys.path.insert(0, str(repo))
    try:
        from video_depth_anything.video_depth import VideoDepthAnything
    except ImportError as exc:  # pragma: no cover - defensive
        raise SystemExit(
            "Could not import VideoDepthAnything. Ensure --repo points to the cloned project."
        ) from exc

    model_cfg = {
        "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    cfg = model_cfg[encoder]
    model = VideoDepthAnything(
        encoder=encoder,
        features=cfg["features"],
        out_channels=cfg["out_channels"],
        metric=metric,
    )
    return model


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    checkpoint = (repo / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint.resolve()
    if not repo.exists():
        raise SystemExit(f"Repository path not found: {repo}")
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    model = load_model(repo, args.encoder, args.metric)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().to(torch.float32)

    dummy = torch.randn(1, 32, 3, args.input_size, args.input_size, dtype=torch.float32)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = {
        "frames": {0: "batch", 1: "time", 3: "height", 4: "width"},
        "depth": {0: "batch", 1: "time", 2: "height", 3: "width"},
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output),
            input_names=["frames"],
            output_names=["depth"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )

    print(f"Exported ONNX model to {output}")


if __name__ == "__main__":
    main()
