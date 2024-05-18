from .detector import Detector
import onnxruntime
from typing import Tuple
import numpy as np

class CornerDetector(Detector):
    def __init__(self, model_path: str = '/home/ayagoz/Documents/projects/MedLafaletCV/detectors/model_weights/cornerDetector.onnx', retain_factor: float = 0.85) -> None:
        super().__init__()
        self.model = onnxruntime.InferenceSession(model_path, providers=self.device)
        self.retain_factor = retain_factor

    def __call__(self, image: np.ndarray) -> Tuple[int]:
        ans_x = 0.0
        ans_y = 0.0
        up_scale_factor = (image.shape[1], image.shape[0])

        while (image.shape[0] > 10 and image.shape[1] > 10):
            img_temp = self.preprocess(image)
            response_up = self.model.run(
                None, {self.model.get_inputs()[0].name: img_temp}
            )[0][0]

            response_up = response_up * up_scale_factor
            x_loc, y_loc = int(response_up[0]), int(response_up[1])

            start_x = min(x_loc + int(round(image.shape[1] * self.retain_factor / 2)), image.shape[1]) - int(round(
                image.shape[1] * self.retain_factor)) if x_loc > image.shape[1] / 2 else max(x_loc - int(
                image.shape[1] * self.retain_factor / 2), 0)
            
            start_y = min(y_loc + int(image.shape[0] * self.retain_factor / 2), image.shape[0]) - int(
                image.shape[0] * self.retain_factor) if y_loc > image.shape[0] / 2 else max(y_loc - int(
                image.shape[0] * self.retain_factor / 2), 0)

            ans_x += start_x
            ans_y += start_y

            image = image[start_y:start_y + int(image.shape[0] * self.retain_factor),
                        start_x:start_x + int(image.shape[1] * self.retain_factor)]
            up_scale_factor = (image.shape[1], image.shape[0])

        ans_x += x_loc
        ans_y += y_loc
        return (int(round(ans_x)), int(round(ans_y)))


        