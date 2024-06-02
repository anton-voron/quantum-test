import torch
from torchvision import datasets, transforms
import random

from digit_classifier import DigitClassifier

transform = transforms.ToTensor()
full_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
random_idx = random.randint(0, len(full_dataset))

images = full_dataset.data
labels = full_dataset.targets

image = images[random_idx]
label = labels[random_idx]
print(f"Image shape: {image.size()}, Label: {label}")

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
