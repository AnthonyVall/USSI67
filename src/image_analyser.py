from utils import render_image, save_image, transform_frame_to_array
from pymba import Vimba, VimbaException, Frame
from pyzbar.pyzbar import decode, ZBarSymbol
from keras.models import load_model
from pathlib import Path
from typing import Tuple
import numpy as np
import time
import cv2


running = False


def run(camera: int = 0,
        size: Tuple[int, int] = (1080, 920),
        model_path: Path = Path('D:\converted_keras/keras_model.h5'),
        labels_path: Path = Path('../labels.txt')
    ) -> None:

    model = load_model(model_path, compile=False)
    class_names = open(str(labels_path), "r").readlines()
    global running
    running = True


    def compute_element(image: np.array):
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
        img = (img / 127.5) - 1

        prediction = model.predict(img)
        index = np.argmax(prediction)

        return class_names[index], prediction[0][index]


    def decode_codes(img: np.array):
        for code in decode(img, symbols=[ZBarSymbol.QRCODE]):
            print(code)


    def handle_frames(frame: Frame):
        img = transform_frame_to_array(frame, size=size)
        render_image(img)

        decode_codes(img)

        class_name, confidence_score = compute_element(img)
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

        key = cv2.waitKey(1)

        if key == 113:  # Q
            print("Closing the application...")
            global running
            running = False


    with Vimba() as vimba:
        camera = vimba.camera(camera)
        camera.open()

        camera.arm('Continuous', handle_frames)
        camera.start_frame_acquisition()

        while running:
            time.sleep(1)

        camera.stop_frame_acquisition()
        camera.disarm()
        camera.close()
