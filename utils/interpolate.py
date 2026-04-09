import cv2

from enum import Enum


class Algorithm(Enum):
    NEAREST_NEIGHBOR = cv2.INTER_NEAREST
    BILINEAR = cv2.INTER_LINEAR
    BICUBIC = cv2.INTER_CUBIC
    LANCZOS = cv2.INTER_LANCZOS4


def interpolate(input: str, output: str, scale: int, algorithm: Algorithm) -> None:
    image = cv2.imread(input)
    interpolated = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=algorithm.value)
    cv2.imwrite(output, interpolated)
