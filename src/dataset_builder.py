from utils import render_image, save_image, transform_frame_to_array
from pymba import Vimba, VimbaException, Frame
from pathlib import Path
from typing import Tuple
import numpy as np
import cv2


def run(camera: int = 0, size: Tuple[int, int] = (1080, 920), save_folder: Path = Path('../save_images')):
    with Vimba() as vimba:
        camera = vimba.camera(camera)
        camera.open()

        feature = camera.feature("ExposureTime")

        camera.arm('SingleFrame')
        i: int = 0

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

            elif k == 115:
                i += 1
                print(f"Saving image number {i}")
                save_image(save_folder / f"image-{i}.jpeg", image)

        camera.disarm()
        camera.close()
