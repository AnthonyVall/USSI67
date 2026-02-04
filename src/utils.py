from sklearn.cluster import KMeans
from typing import Tuple
from pathlib import Path
from pymba import Frame
import pandas as pd
import numpy as np
import math
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


def compute_pixel_size(image: np.array, rect_size: Tuple[float, float]) -> Tuple[float, float]:
    img = cv2.Canny(image, 200, 230)
    lines = cv2.HoughLinesP(img, rho=1, theta=np.pi / 180, threshold=100, minLineLength=250, maxLineGap=100)
    lines = [[line[0][0], line[0][1], line[0][2], line[0][3], math.degrees(math.atan2(line[0][3] - line[0][1], line[0][2] - line[0][0]))]  for line in lines]

    # TODO have angle more important ?
    data: pd.DataFrame = pd.DataFrame(lines, columns=["x1", "y1", "x2", "y2", "angle"])
    init = np.array([
        [540, 0, 540, 0, -90],
        [540, 920, 540, 920, -90],
        [0, 460, 0, 460, 0],
        [1080, 460, 1080, 460, 0]
    ])
    kmeans = KMeans(n_clusters=4, init=init)
    kmeans.fit(data)

    l = dict()
    data['group'] = kmeans.labels_
    for label in range(4):
        d = data[data['group'] == label]
        l[label] = ((min(d['x1']), min(d['y1'])), (max(d['x2']), max(d['y2'])))

    print(l)
    return ""


if "__main__" == __name__:
    img = cv2.imread('../data/square.png')

    dst = np.copy(img)
    print(compute_pixel_size(img, rect_size=(170, 120)))

    """
    img  = cv2.Canny(img, 200, 230)
    lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=100, minLineLength=250, maxLineGap=100)

    lines: list[Tuple[Tuple[int, int], Tuple[int, int]]] = [((line[0][0], line[0][1]), (line[0][2], line[0][3])) for line in lines]
    for line in lines:
        cv2.line(dst, line[0], line[1], (255, 0, 0), 2)

    cv2.imshow('img', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    # (2.911764705882353, 4.966666666666667)
