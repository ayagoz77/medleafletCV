import cv2
from typing import Tuple, List
import numpy as np

class TextExtractor:
    def __init__(self, rect_size: Tuple[int] = (5, 5)) -> None:
        self.rect_size = rect_size

    def __call__(self, image: np.ndarray) -> List[Tuple[int]]:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('debug_imgs/gray.png', gray)
        ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        cv2.imwrite('debug_imgs/thrsh.png', thresh1)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.rect_size)
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
        cv2.imwrite('debug_imgs/dilation.png', dilation)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        sorted_rects = sorted((cv2.boundingRect(cnt) for cnt in contours), key=lambda rect: (rect[0], rect[1]))

        return sorted_rects