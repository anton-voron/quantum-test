import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

from .digit_classification_interface import DigitClassificationInterface

class RFModel(DigitClassificationInterface):
    def __init__(self):

        model_path = os.path.join(os.getcwd(), 'model_weights', 'rf_model.pkl')
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)  
            print("Loaded RandomForest model weights")
        else:
            print("RandomForest model weights not found")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def predict(self, image):
        image = image.numpy().reshape(-1, 28*28)
        return self.model.predict(image)[0]
