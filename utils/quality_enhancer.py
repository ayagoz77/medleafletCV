from PIL import Image, ImageEnhance
import numpy as np

class QualityEnhancer:
    def __init__(self, brightness=1.7, contrast=2) -> None:
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image: np.ndarray) -> np.ndarray:
        img = Image.fromarray(image)
        
        # Adjust brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(self.brightness)
        
        # Adjust contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.contrast)
        
        return np.array(img)