import argparse
import piq
import torch

from models.srcnn import SRCNN
from pathlib import Path
from torchvision.utils import save_image
from utils.div2k_2018_dataset import Div2k2018TestDataset


def run_test(model, test_loader, scale: int, save_folder: Path) -> tuple[float, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_psnr = 0.0
    total_ssim = 0.0

    model.eval()
    with torch.no_grad():
        for i, (lr_image, ground_truth_hr_image) in enumerate(test_loader):
            lr_image = lr_image.to(device)
            ground_truth_hr_image = ground_truth_hr_image.to(device)

            # Upscale LR image
            lr_upscaled = torch.nn.functional.interpolate(
                lr_image, scale_factor=scale, mode="bicubic", align_corners=False
            ).clamp(0, 1)

            # Inference
            hr_output = model(lr_upscaled).clamp(0, 1)

            # Compute metrics
            psnr = piq.psnr(hr_output, ground_truth_hr_image, data_range=1.0)
            ssim = piq.ssim(hr_output, ground_truth_hr_image, data_range=1.0)
            total_psnr += psnr.item()
            total_ssim += ssim.item()

            save_image(hr_output, save_folder / f"{i:04d}.png")
            print(f"Image {i}: PSNR={psnr.item():.6f}")

    count = len(test_loader)
    return (total_psnr / count, total_ssim / count)


def test_srcnn(scale: int, model_path: str, output_dir: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Testing dataset
    test_dataset = Div2k2018TestDataset(scale=scale)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda")
    )

    # Load model
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model_name = Path(model_path).stem
    save_folder = Path(output_dir) / model_name
    save_folder.mkdir(parents=True, exist_ok=True)

    average_psnr, average_ssim = run_test(model=model, test_loader=test_loader, scale=scale, save_folder=save_folder)
    print(f"Average PSNR={average_psnr}, Average SSIM={average_ssim}")


def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog="Testing ISR", description="Testing script for image super-resolution using ML"
    )
    parser.add_argument("model", type=str, choices=["srcnn"], help="Selected ML model")
    parser.add_argument("-s", "--scale", type=int, default=4, help="Upscale factor")
    parser.add_argument(
        "-mp", "--model-path", type=str, required=True, help="Path to .pth file (must align with scale)"
    )
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    print(f"Testing script")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Is CUDA available: {torch.cuda.is_available()}")

    if args.model == "srcnn":
        test_srcnn(
            scale=args.scale,
            model_path=args.model_path,
            output_dir=args.output_dir,
        )
    else:
        print("Invalid model")

    print("Finished testing")


if __name__ == "__main__":
    main()
