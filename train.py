import argparse
import torch
import os

from models.rlsp import RLSP
from models.srcnn import SRCNN
from models.vdsr import VDSR
from utils.div2k_2018_dataset import Div2k2018TrainDataset
from utils.reds_dataset import RedsDataset


def train_vdsr(
    scale: int, patch_size: int, batch_size: int, epochs: int, save_path: str | None, resume_path: str | None = None
) -> None:
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 1e-4  # high learning rate for VDSR

    train_dataset = Div2k2018TrainDataset(scale=scale, patch_size=patch_size, scale_variant=scale)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda")
    )

    vdsr = VDSR().to(device)

    if resume_path:
        print(f"Loading previous weights from: {resume_path}")
        vdsr.load_state_dict(torch.load(resume_path, map_location=device, weights_only=True))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(vdsr.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(epochs):
        vdsr.train()
        epoch_loss = 0.0

        for i, (lr_image, ground_truth_hr_image) in enumerate(train_loader, 0):
            lr_image = lr_image.to(device, non_blocking=True)
            ground_truth_hr_image = ground_truth_hr_image.to(device, non_blocking=True)

            # Upscale lr image (Data Pipeline requirement)
            lr_image = torch.nn.functional.interpolate(
                lr_image, scale_factor=scale, mode="bicubic", align_corners=False
            ).clamp(0, 1)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                output_hr_image = vdsr(lr_image)
                loss = criterion(output_hr_image, ground_truth_hr_image)

            scaler.scale(loss).backward()

            # Gradient Clipping: clip at 0.1 / lr
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(vdsr.parameters(), 0.1 / lr)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}: batch_loss={loss.item():.6f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}: average_loss={avg_epoch_loss:.6f}")
        torch.save(vdsr.state_dict(), f"{f'{save_path}/' if save_path else ''}vdsr_{scale}x_e{epoch}.pth")


def train_srcnn(
    scale: int, patch_size: int, batch_size: int, epochs: int, save_path: str | None, resume_path: str | None = None
) -> None:
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training dataset
    train_dataset = Div2k2018TrainDataset(scale=scale, patch_size=patch_size, scale_variant=scale)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda")
    )

    # ML model
    srcnn = SRCNN().to(device)

    if resume_path:
        print(f"Loading previous weights from: {resume_path}")
        srcnn.load_state_dict(torch.load(resume_path, map_location=device, weights_only=True))

    # Loss function
    criterion = torch.nn.MSELoss()
    # Optimization algorithm
    optimizer = torch.optim.Adam(
        [
            {"params": srcnn.conv1.parameters(), "lr": 1e-4},
            {"params": srcnn.conv2.parameters(), "lr": 1e-4},
            {"params": srcnn.conv3.parameters(), "lr": 1e-5},
        ]
    )

    # Training
    for epoch in range(epochs):
        srcnn.train()
        epoch_loss = 0.0

        for i, (lr_image, ground_truth_hr_image) in enumerate(train_loader, 0):
            lr_image = lr_image.to(device, non_blocking=True)
            ground_truth_hr_image = ground_truth_hr_image.to(device, non_blocking=True)

            # Upscale lr image
            lr_image = torch.nn.functional.interpolate(
                lr_image, scale_factor=scale, mode="bicubic", align_corners=False
            )
            lr_image = torch.clamp(lr_image, 0, 1)

            # Zero out the parameter gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward
            output_hr_image = srcnn(lr_image)
            # Loss
            loss = criterion(output_hr_image, ground_truth_hr_image)
            # Back propagation
            loss.backward()
            # Optimize
            optimizer.step()

            # Print statistics
            epoch_loss += loss.item()
            print(f"Epoch {epoch}, Batch {i}: batch_loss={loss.item():.6f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}: average_loss={avg_epoch_loss:.6f}")

        torch.save(srcnn.state_dict(), f"{f'{save_path}/' if save_path else ''}srcnn_{scale}x_e{epoch}.pth")


def train_rlsp(
    patch_size: int, batch_size: int, epochs: int, save_path: str | None, resume_path: str | None = None
) -> None:
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training dataset
    train_dataset = RedsDataset(root_dir="data/train/reds/train/train_sharp", patch_size=patch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda")
    )

    # ML model
    rlsp = RLSP().to(device)

    if resume_path:
        print(f"Loading previous weights from: {resume_path}")
        rlsp.load_state_dict(torch.load(resume_path, map_location=device, weights_only=True))

    # Loss function
    criterion = torch.nn.MSELoss()
    # Optimization algorithm
    optimizer = torch.optim.Adam(rlsp.parameters(), lr=1e-4)

    # Training
    for epoch in range(epochs):
        rlsp.train()
        epoch_loss = 0.0

        for i, (lr_seq, hr_seq) in enumerate(train_loader):
            # Inputs: [B, T, 3, H, W]
            lr_seq = lr_seq.to(device)
            hr_seq = hr_seq.to(device)

            optimizer.zero_grad()

            # Forward
            output_seq = rlsp(lr_seq)

            # Loss
            loss = criterion(output_seq, hr_seq)

            # Back propagation
            loss.backward()

            # Optimize
            optimizer.step()

            # Print statistics
            epoch_loss += loss.item()
            print(f"Epoch {epoch}, Batch {i}: batch_loss={loss.item():.6f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}: average_loss={avg_epoch_loss:.6f}")

        torch.save(rlsp.state_dict(), f"{f'{save_path}/' if save_path else ''}rlsp_4x_e{epoch}.pth")


def main() -> None:
    torch.backends.cudnn.benchmark = True
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog="Training ISR", description="Training script for image super-resolution using ML"
    )
    parser.add_argument("model", type=str, choices=["srcnn", "vdsr", "rlsp"], help="Selected ML model")
    parser.add_argument("-s", "--scale", type=int, default=2, help="Upscale factor")
    parser.add_argument("-p", "--patch-size", type=int, default=64, help="Image size used for training")
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("-sp", "--save-path", type=str, help="Directory where to save trained NN")
    parser.add_argument("-r", "--resume", type=str, default=None, help="Path to .pth file to resume training")

    args = parser.parse_args()

    print(f"Started training")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Is CUDA available: {torch.cuda.is_available()}")

    # Train selected model
    if args.model == "srcnn":
        train_srcnn(
            scale=args.scale,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            save_path=args.save_path,
            resume_path=args.resume,
        )
    elif args.model == "vdsr":
        train_vdsr(
            scale=args.scale,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            save_path=args.save_path,
            resume_path=args.resume,
        )
    elif args.model == "rlsp":
        train_rlsp(
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            save_path=args.save_path,
            resume_path=args.resume,
        )
    else:
        print("Invalid model")

    print("Finished training")


if __name__ == "__main__":
    main()
