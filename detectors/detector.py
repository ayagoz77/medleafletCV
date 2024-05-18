from abc import ABC, abstractmethod
from typing import Any
import cv2
import numpy as np

class Detector(ABC):
    def __init__(self) -> None:
        self.device = [("CPUExecutionProvider")]
    
    def preprocess(self, image: np.ndarray):

        img = cv2.resize(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB).astype(np.float32), (32, 32))
        img = np.expand_dims(img.transpose(2, 0, 1), axis=0)

        return img
    
    @abstractmethod
    def __call__(self, image: np.ndarray) -> Any:
        pass