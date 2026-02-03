from utils import save_image
from pathlib import Path
import pandas as pd
import numpy as np

class BielletteSaver(object):

    def __init__(self, csv_file: Path, images_folder: Path, columns: str = ["items"]):
        self.images_folder: Path = images_folder
        images_folder.mkdir(parents=True, exist_ok=True)
        self.columns = columns

        self.csv_file: Path = csv_file
        if csv_file.is_file():
            self.biellettes: pd.DataFrame = pd.read_csv(str(csv_file), index_col = 0, header = 0)
        else:
            self.biellettes: pd.DataFrame = pd.DataFrame(columns = self.columns)


    def __contains__(self, item) -> bool:
        return (self.biellettes == item).all(1).any()


    def add_biellette(self, item, img: np.array):
        if not self.__contains__(item):
            self.biellettes = pd.concat([self.biellettes, pd.DataFrame([item], columns = self.columns)], ignore_index=True)
            id: int = len(self.biellettes) - 1
            save_image(img, self.images_folder / f"{id}.png")
            print(f'Biellette {id} added with value {item}')


    def save_biellettes(self, csv_file: Path = None) -> None:
        if csv_file is None:
            csv_file = self.csv_file

        pd.DataFrame(self.biellettes).to_csv(str(csv_file), index=True)
