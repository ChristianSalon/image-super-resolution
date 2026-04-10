import argparse

from interpolate import interpolate, Algorithm
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Initialize dataset", description="Create LR from HR images using selected interpolation algorithm"
    )
    parser.add_argument("-i", "--input", type=str, help="HR images directory")
    parser.add_argument("-o", "--output", type=str, help="LR images directory")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    for file in input_path.iterdir():
        if file.is_file():
            path = output_path / "lr_nn_2x" / file.name
            interpolate(input=file.absolute(), output=path, scale=0.5, algorithm=Algorithm.NEAREST_NEIGHBOR)

            path = output_path / "lr_nn_4x" / file.name
            interpolate(input=file.absolute(), output=path, scale=0.25, algorithm=Algorithm.NEAREST_NEIGHBOR)

            path = output_path / "lr_nn_8x" / file.name
            interpolate(input=file.absolute(), output=path, scale=0.125, algorithm=Algorithm.NEAREST_NEIGHBOR)

            path = output_path / "lr_bilin_2x" / file.name
            interpolate(input=file.absolute(), output=path, scale=0.5, algorithm=Algorithm.BILINEAR)

            path = output_path / "lr_bilin_4x" / file.name
            interpolate(input=file.absolute(), output=path, scale=0.25, algorithm=Algorithm.BILINEAR)

            path = output_path / "lr_bilin_8x" / file.name
            interpolate(input=file.absolute(), output=path, scale=0.125, algorithm=Algorithm.BILINEAR)

            path = output_path / "lr_bicub_2x" / file.name
            interpolate(input=file.absolute(), output=path, scale=0.5, algorithm=Algorithm.BICUBIC)

            path = output_path / "lr_bicub_4x" / file.name
            interpolate(input=file.absolute(), output=path, scale=0.25, algorithm=Algorithm.BICUBIC)

            path = output_path / "lr_bicub_8x" / file.name
            interpolate(input=file.absolute(), output=path, scale=0.125, algorithm=Algorithm.BICUBIC)

            path = output_path / "lr_lan_2x" / file.name
            interpolate(input=file.absolute(), output=path, scale=0.5, algorithm=Algorithm.LANCZOS)

            path = output_path / "lr_lan_4x" / file.name
            interpolate(input=file.absolute(), output=path, scale=0.25, algorithm=Algorithm.LANCZOS)

            path = output_path / "lr_lan_8x" / file.name
            interpolate(input=file.absolute(), output=path, scale=0.125, algorithm=Algorithm.LANCZOS)


if __name__ == "__main__":
    main()
