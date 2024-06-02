import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .digit_classification_interface import DigitClassificationInterface

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNNModel(DigitClassificationInterface):
    def __init__(self):
        self.model = SimpleCNN()
        model_path = os.path.join(os.getcwd(), 'model_weights', 'cnn_model.pth')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print("Loaded CNN model weights")
        else:
            print("CNN model weights not found")

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            
        self.model.eval()

    def predict(self, image):
        image = image.float() / 255.0
        with torch.no_grad():
            if image.dim() == 2:  
                image = image.unsqueeze(0).unsqueeze(0)  
            elif image.dim() == 3:  
                image = image.unsqueeze(0)  
            if torch.cuda.is_available():
                image = image.cuda()
            output = self.model(image)
            _, predicted = torch.max(output, 1)
            return predicted.item()
