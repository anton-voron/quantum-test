import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import numpy as np

from models import SimpleCNN


def train_cnn():
    transform = transforms.ToTensor()
    full_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    train_indices, test_indices = train_test_split(np.arange(len(full_dataset)), test_size=0.2, random_state=42)
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 11):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch}/10], Loss: {avg_loss:.4f}')

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

    model_path = os.path.join(os.getcwd(), 'model_weights')
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), f'{model_path}/cnn_model.pth')


if __name__ == '__main__':
    train_cnn()