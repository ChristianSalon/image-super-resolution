import argparse
import numpy as np
import torch

from models.srcnn import SRCNN
from torchvision.io import decode_image, ImageReadMode
from torchvision.utils import save_image


def run_inference(model, lr_image_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    with torch.no_grad():
        output = model(lr_image_tensor)
        # Delete batch dimension [C, H, W]
        output = output.squeeze(0).cpu().clamp(0, 1)

        return output


def run_srcnn(scale: int, model_path: str, input_path: str, output_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr_image = decode_image(input_path, mode=ImageReadMode.RGB).float() / 255.0
    # Add batch dimension [1, C, H, W]
    lr_image = lr_image.unsqueeze(0).to(device)
    # Upscale input lr image
    lr_image = torch.nn.functional.interpolate(lr_image, scale_factor=scale, mode="bicubic", align_corners=False)

    # Load model
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    output = run_inference(model=model, lr_image_tensor=lr_image)

    save_image(output, fp=output_path)


def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog="Inference ISR", description="Inference script for image super-resolution using ML"
    )
    parser.add_argument("model", type=str, choices=["srcnn"], help="Selected ML model")
    parser.add_argument("-s", "--scale", type=int, default=4, help="Upscale factor")
    parser.add_argument(
        "-mp", "--model-path", type=str, required=True, help="Path to .pth file (must align with scale)"
    )
    parser.add_argument("-i", "--input-path", type=str, required=True, help="Path to input LR image")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Path to output HR image")

    args = parser.parse_args()

    print(f"Started inference")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Is CUDA available: {torch.cuda.is_available()}")

    # Train selected model
    if args.model == "srcnn":
        run_srcnn(
            scale=args.scale,
            model_path=args.model_path,
            input_path=args.input_path,
            output_path=args.output_path,
        )
    else:
        print("Invalid model")

    print("Finished inference")


if __name__ == "__main__":
    main()
