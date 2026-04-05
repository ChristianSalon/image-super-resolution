import argparse
import torch

from models.srcnn import SRCNN
from utils.div2k_2018_dataset import Div2k2018TrainDataset


def train_srcnn(scale: int, patch_size: int, batch_size: int) -> None:
    # Training dataset
    train_dataset = Div2k2018TrainDataset(scale=scale, patch_size=patch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # ML model
    srcnn = SRCNN()

    # Loss function
    criterion = torch.nn.MSELoss()
    # Optimization algorithm
    optimizer = torch.optim.Adam(srcnn.parameters(), lr=0.001)

    # Training
    for epoch in range(100):
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            lr_image, ground_truth_hr_image = data

            # Upscale lr image
            lr_image = torch.nn.functional.interpolate(
                lr_image, scale_factor=scale, mode="bicubic", align_corners=False
            )

            # Zero out the parameter gradients
            optimizer.zero_grad()

            # Forward
            output_hr_image = srcnn(lr_image)
            # Loss
            loss = criterion(output_hr_image, ground_truth_hr_image)
            # Back propagation
            loss.backward()
            # Optimize
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0


def main() -> None:
    print(f"Started training")

    # Parse arguments
    parser = argparse.ArgumentParser(
        prog="Training ISR", description="Training script for image super-resolution using ML"
    )
    parser.add_argument("model")
    parser.add_argument("-s", "--scale", type=int, default=4)
    parser.add_argument("-p", "--patch-size", type=int, default=64)
    parser.add_argument("-b", "--batch-size", type=int, default=8)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Is CUDA available: {torch.cuda.is_available()}")

    # Train selected model
    args = parser.parse_args()
    if args.model == "srcnn":
        train_srcnn(args.scale, args.patch_size, args.batch_size)
    else:
        print("Invalid model")

    print("Finished training")


if __name__ == "__main__":
    main()
