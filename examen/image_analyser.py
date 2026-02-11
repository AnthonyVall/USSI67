from utils import render_image, transform_frame_to_array
from pymba import Vimba, VimbaException, Frame
from scipy.spatial import distance as dist
from keras.models import load_model
from pathlib import Path
from typing import Tuple
import numpy as np
import time
import cv2

running = False


def run(camera: int = 0,
        size: Tuple[int, int] = (1080, 920),
        model_path: Path = Path('D:\\converted_keras\\vis\\keras_model.h5'),
        labels_path: Path = Path('D:\\converted_keras\\vis\\labels.txt')
    ) -> None:

    model = load_model(model_path, compile=False)
    class_names = open(str(labels_path), "r").readlines()
    global running
    running = True

    compteur: dict[str, int] = {}

    def compute_element(image: np.array):
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
        img = (img / 127.5) - 1

        prediction = model.predict(img)
        index = np.argmax(prediction)

        return class_names[index], prediction[0][index]


    def midpoint(ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


    def compute_vis_size(img: np.array) -> Tuple[float, float, np.array]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        cnts, hierarchy = cv2.findContours(thresh, 1, 2)

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 300 or area > 10000:
                continue

            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            cv2.drawContours(img, [box.astype("int")], -1, (0, 255, 0), 2)

            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            return dA, dB, img

        return -1, -1, img


    def compute_vis_cat(curent_size: float, grande_vis: float = 215, moyenne_vis: float = 140, petite_vis: float = 75) -> int:
        dists = np.abs(np.array([grande_vis, moyenne_vis, petite_vis]) - np.array([curent_size, curent_size, curent_size]))
        return dists.argmin()


    def get_vis_type_from_cat(cat: int) -> str:
        if cat == 0:
            return "grande vis"
        elif cat == 1:
            return "moyenne vis"
        elif cat == 2:
            return "petite vis"

        return "unknown"


    def handle_frames(frame: Frame, grande_vis: float = 215, moyenne_vis: float = 140, petite_vis: float = 75):
        img = transform_frame_to_array(frame, size=size)

        class_name, confidence_score = compute_element(img)
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

        if class_name[2:] == "vis\n":
            dA, dB, img = compute_vis_size(img)
            cat = compute_vis_cat(max(dA, dB), grande_vis, moyenne_vis, petite_vis)
            cat_name = get_vis_type_from_cat(cat)
            print(f"Il s'agit d'une {cat_name}")

        render_image(img)

        key = cv2.waitKey(1)

        if key == 113:  # Q
            print("Closing the application...")
            global running
            running = False
            cv2.destroyAllWindows()
            print(f"Vis capturées: {compteur}")

        elif key == 112: # P
            if class_name[2:] == "vis\n":
                dA, dB, img = compute_vis_size(img)
                cat = compute_vis_cat(max(dA, dB), grande_vis, moyenne_vis, petite_vis)
                cat_name = get_vis_type_from_cat(cat)
                print(f"Capturing {cat_name}")
                compteur[cat_name] = compteur.get(cat_name, 0) + 1


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


if __name__ == "__main__":
    run()
