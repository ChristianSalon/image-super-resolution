import torch


def main():
    print(f"Training script")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Is CUDA available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
