import numpy as np

from .digit_classification_interface import DigitClassificationInterface

class RandModel(DigitClassificationInterface):
    def __init__(self):
        pass

    def predict(self, image):
        if not isinstance(image, np.ndarray):
            image = image.numpy()
        
        center_crop = image[9:19, 9:19]
        
        random_prediction = np.random.randint(0, 10)
        return random_prediction
