from models import CNNModel, RFModel, RandModel

class DigitClassifier:
    def __init__(self, algorithm):
        if algorithm == 'cnn':
            self.model = CNNModel()
        elif algorithm == 'rf':
            self.model = RFModel()
        elif algorithm == 'rand':
            self.model = RandModel()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def predict(self, image):
        return self.model.predict(image)
