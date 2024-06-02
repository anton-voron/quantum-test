import torch
from digit_classifier import DigitClassifier

# Load a sample image (28x28) for testing
image = torch.rand(28, 28)  # Replace with actual MNIST image

# Test CNN Model
cnn_classifier = DigitClassifier('cnn')
cnn_prediction = cnn_classifier.predict(image)
print(f"CNN Prediction: {cnn_prediction}")

# Test Random Forest Model
rf_classifier = DigitClassifier('rf')
rf_prediction = rf_classifier.predict(image)
print(f"RF Prediction: {rf_prediction}")

# Test Random Value Model
rand_classifier = DigitClassifier('rand')
rand_prediction = rand_classifier.predict(image)
print(f"Random Model Prediction: {rand_prediction}")
