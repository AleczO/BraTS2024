"""3D U-Net for BraTS 2024 GLI segmentation (MONAI implementation).

Input:  (B, 4, X, Y, Z)  - modalities t1n, t1c, t2w, t2f
Output: (B, C, X, Y, Z)  - raw logits, one channel per region:
        ET, TC, WT and, if include_rc, RC (see src/dataset/transforms.py)

Regions overlap (ET ⊂ TC ⊂ WT), so each channel is an independent yes/no
decision: use sigmoid (not softmax) on the logits, e.g. DiceLoss(sigmoid=True),
and threshold each channel at 0.5 for the final mask.

Quick check (shape, parameters, GPU memory for one forward/backward pass):
    python -m src.model
    python -m src.model --patch 128 --batch 2
"""

import argparse

import torch
from monai.networks.nets import UNet

IN_CHANNELS = 4  # t1n, t1c, t2w, t2f


def build_unet(
    include_rc: bool = True,
    channels: tuple[int, ...] = (32, 64, 128, 256, 320),
    strides: tuple[int, ...] = (2, 2, 2, 2),
    num_res_units: int = 2,
    dropout: float = 0.0,
) -> UNet:
    """MONAI 3D U-Net with residual units and instance normalisation.

    channels: feature maps per level, from the highest resolution to the bottleneck.
    strides:  downsampling factor between levels (one fewer than channels).
              With four strides of 2, input sizes must be divisible by 2**4 = 16.
    """
    return UNet(
        spatial_dims=3,
        in_channels=IN_CHANNELS,
        out_channels=4 if include_rc else 3,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        norm="instance",
        dropout=dropout,
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the U-Net and run one forward/backward pass.")
    parser.add_argument("--patch", type=int, default=128, help="Cubic patch size (divisible by 16)")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for the test pass")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet().to(device)
    print(f"Parameters: {count_parameters(model) / 1e6:.1f} M")
    print(f"Device:     {device}")

    x = torch.randn(args.batch, IN_CHANNELS, args.patch, args.patch, args.patch, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    out = model(x)
    out.mean().backward()  # include gradients in the memory measurement
    print(f"Input:      {tuple(x.shape)}")
    print(f"Output:     {tuple(out.shape)}")
    if device.type == "cuda":
        print(f"Peak GPU memory (fp32, forward + backward): "
              f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()