from utils import render_image, save_image, transform_frame_to_array
from pymba import Vimba, VimbaException, Frame
from pathlib import Path
from typing import Tuple
import numpy as np
import random as rd
import cv2


def run(camera: int = 0, size: Tuple[int, int] = (1080, 920), save_folder: Path = Path('./img')):
    with Vimba() as vimba:
        camera = vimba.camera(camera)
        camera.open()

        feature = camera.feature("ExposureTime")

        camera.arm('SingleFrame')
        i: int = 33

        image = np.zeros(size)
        render_image(image)

        while True:
            k = cv2.waitKey(33)
            if k == 113:  # Q
                print("Closing the application...")
                break

            elif k == 112:  # P
                print("Taking picture...")
                image: np.array = transform_frame_to_array(camera.acquire_frame(), size=size)
                render_image(image)

            elif k == 111:  # O
                feature.value += 10000
                print(f'Changing ExposureTime to {feature.value}')

            elif k == 108:  # L
                feature.value -= 10000
                print(f'Changing ExposureTime to {feature.value}')

            elif k == 115: # S
                i += 1
                print(f"Saving image number {i}")
                save_image(image, save_folder / f"image-{i}.jpeg")

        camera.disarm()
        camera.close()

def edit_brightness(img: np.array, add: int = 0, reduce: int = 0) -> np.array:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if add > 0:
        lim_top = 255 - add
        v[v > lim_top] = 255
        v[v <= lim_top] += add

    if reduce > 0:
        lim_bot = 0 + reduce
        v[v < lim_bot] = 0
        v[v >= lim_bot] -= reduce

    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def edit_contrast(img: np.array, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.array:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def edit_saturation(img: np.array, sat: float = 5) -> np.array:
    imghsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(imghsv)
    s = s * sat
    s = np.clip(s, 0, 255)
    imghsv = cv2.merge([h, s, v])
    return cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)


def smoothing(img: np.array, kernel: np.array = np.ones((5,5),np.float32) / 25) -> np.array:
    return cv2.filter2D(img, -1, kernel)


def augmente_images(input_images: Path, output_images: Path, n_images: int = 10) -> None:
    for image_path in input_images.glob("*.jpeg"):
        image = cv2.imread(str(image_path))
        for i in range(n_images):
            img = np.copy(image)

            if rd.choice([True, False]):
                img = edit_brightness(img, add=rd.randint(0, 255), reduce=rd.randint(0, 255))

            if rd.choice([True, False]):
                img = edit_contrast(img, clip_limit=float(rd.randint(0, 10000)) / 100, tile_grid_size=(rd.randint(5, 100), rd.randint(5, 100)))

            if rd.choice([True, False]):
                img = edit_saturation(img, sat=float(rd.randint(0, 10000)) / 100)

            if rd.choice([True, False]):
                k: int = rd.randint(5, 20)
                img = smoothing(img, kernel=np.ones((k,k),np.float32) / (k*k))

            save_image(img, output_images / f'augmented_{image_path.name}_nb_{i}.jpeg')


if __name__ == '__main__':
    #run()
    #render_image(edit_brightness(cv2.imread(str(Path("./img/image-1.jpeg"))), reduce=200))
    #render_image(edit_contrast(cv2.imread(str(Path("./img/image-1.jpeg"))), tile_grid_size=(50, 50)))
    augmente_images(Path('./img'), Path('./augmented_img'), n_images=20)
