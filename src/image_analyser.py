from utils import render_image, save_image, transform_frame_to_array, compute_pixel_size
from pymba import Vimba, VimbaException, Frame
from pyzbar.pyzbar import decode, ZBarSymbol
from keras.models import load_model
from pathlib import Path
from typing import Tuple
from BielletteSaver import BielletteSaver
import numpy as np
import time
import cv2


running = False
pixel_size = -1


def run(camera: int = 0,
        size: Tuple[int, int] = (1080, 920),
        model_path: Path = Path('D:\converted_keras/keras_model.h5'),
        labels_path: Path = Path('D:\converted_keras/labels.txt'),
        data_path: Path = Path('../data'),

    ) -> None:

    model = load_model(model_path, compile=False)
    class_names = open(str(labels_path), "r").readlines()
    bielletteSaver: BielletteSaver = BielletteSaver(data_path / "biellettes.csv", data_path / "images")
    global running
    running = True


    def compute_element(image: np.array):
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
        img = (img / 127.5) - 1

        prediction = model.predict(img)
        index = np.argmax(prediction)

        return class_names[index], prediction[0][index]


    def decode_codes(img: np.array) -> np.array:
        return_img: np.array = img
        for code in decode(img, symbols=[ZBarSymbol.QRCODE]):
            return_img = cv2.polylines(img, [np.array([code.polygon], np.int32)], True, (0, 255, 0), 3)
            bielletteSaver.add_biellette(code.data, img)




        return return_img


    def handle_frames(frame: Frame):
        img = transform_frame_to_array(frame, size=size)

        """
        class_name, confidence_score = compute_element(img)
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
        """

        img = decode_codes(img)
        render_image(img)

        key = cv2.waitKey(1)

        if key == 113:  # Q
            print("Closing the application...")
            bielletteSaver.save_biellettes()
            global running
            running = False
            cv2.destroyAllWindows()


    with Vimba() as vimba:
        camera = vimba.camera(camera)
        camera.open()

        camera.arm('SingleFrame')
        global pixel_size
        pixel_size = compute_pixel_size(transform_frame_to_array(camera.acquire_frame(), size=size), rect_size = (170, 120))
        camera.disarm()

        camera.arm('Continuous', handle_frames)
        camera.start_frame_acquisition()

        while running:
            time.sleep(1)

        camera.stop_frame_acquisition()
        camera.disarm()
        camera.close()
