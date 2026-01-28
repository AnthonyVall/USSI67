from typing import Tuple
from pathlib import Path
from pymba import Frame
import numpy as np
import cv2

PIXEL_FORMATS_CONVERSIONS = {
    'BayerRG8': cv2.COLOR_BAYER_RG2RGB,
}


def transform_frame_to_array(frame: Frame, size: Tuple[int, int] = (1080, 920)) -> np.ndarray:
    image = cv2.resize(frame.buffer_data_numpy(), size)

    try:
        image = cv2.cvtColor(image, PIXEL_FORMATS_CONVERSIONS[frame.pixel_format])
    except KeyError:
        pass

    return image


def render_image(img: np.array, window_name: str = "Window") -> None:
    cv2.imshow(window_name, img)


def save_image(img: np.array, path: Path) -> None:
    cv2.imwrite(str(path), img)
