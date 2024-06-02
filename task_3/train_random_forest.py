import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
import joblib
import os

train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

X = train_dataset.data.numpy().reshape(-1, 28*28)
y = train_dataset.targets.numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Model Accuracy: {accuracy:.4f}')

model_path = os.path.join(os.getcwd(), 'model_weights')
os.makedirs(model_path, exist_ok=True)
joblib.dump(rf_model, os.path.join(model_path, 'rf_model.pkl'))
print(f'Model saved at {model_path}/rf_model.pkl')