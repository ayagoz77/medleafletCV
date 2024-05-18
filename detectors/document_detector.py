from .detector import Detector
import onnxruntime
from typing import Tuple
import numpy as np

class DocumentDetector(Detector):
    def __init__(self, model_path: str = '/home/ayagoz/Documents/projects/MedLafaletCV/detectors/model_weights/docDetector.onnx') -> None:
        super().__init__()
        self.model = onnxruntime.InferenceSession(model_path, providers=self.device)

    def __call__(self, image: np.ndarray) -> Tuple[Tuple[np.ndarray]]:
        img = self.preprocess(image)

        prediction = self.model.run(
                None, {self.model.get_inputs()[0].name: img}
            )[0][0]
        
        x_cords = prediction[[0, 2, 4, 6]]
        y_cords = prediction[[1, 3, 5, 7]]

        x_cords = x_cords * image.shape[1]
        y_cords = y_cords * image.shape[0]

        top_left = image[
                    max(0, int(2 * y_cords[0] - (y_cords[3] + y_cords[0]) / 2)):int((y_cords[3] + y_cords[0]) / 2),
                    max(0, int(2 * x_cords[0] - (x_cords[1] + x_cords[0]) / 2)):int((x_cords[1] + x_cords[0]) / 2)]

        top_right = image[
                    max(0, int(2 * y_cords[1] - (y_cords[1] + y_cords[2]) / 2)):int((y_cords[1] + y_cords[2]) / 2),
                    int((x_cords[1] + x_cords[0]) / 2):min(image.shape[1] - 1,
                                                            int(x_cords[1] + (x_cords[1] - x_cords[0]) / 2))]

        bottom_right = image[int((y_cords[1] + y_cords[2]) / 2):min(image.shape[0] - 1, int(
            y_cords[2] + (y_cords[2] - y_cords[1]) / 2)),
                        int((x_cords[2] + x_cords[3]) / 2):min(image.shape[1] - 1,
                                                                int(x_cords[2] + (x_cords[2] - x_cords[3]) / 2))]

        bottom_left = image[int((y_cords[0] + y_cords[3]) / 2):min(image.shape[0] - 1, int(
            y_cords[3] + (y_cords[3] - y_cords[0]) / 2)),
                        max(0, int(2 * x_cords[3] - (x_cords[2] + x_cords[3]) / 2)):int(
                            (x_cords[3] + x_cords[2]) / 2)]

        top_left = (top_left, max(0, int(2 * x_cords[0] - (x_cords[1] + x_cords[0]) / 2)),
                    max(0, int(2 * y_cords[0] - (y_cords[3] + y_cords[0]) / 2)))
        top_right = (
        top_right, int((x_cords[1] + x_cords[0]) / 2), max(0, int(2 * y_cords[1] - (y_cords[1] + y_cords[2]) / 2)))
        bottom_right = (bottom_right, int((x_cords[2] + x_cords[3]) / 2), int((y_cords[1] + y_cords[2]) / 2))
        bottom_left = (bottom_left, max(0, int(2 * x_cords[3] - (x_cords[2] + x_cords[3]) / 2)),
                        int((y_cords[0] + y_cords[3]) / 2))

        return top_left, top_right, bottom_right, bottom_left

