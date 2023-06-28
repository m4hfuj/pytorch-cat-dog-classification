import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from dataset import train_loader, test_loader
from model import ConvNeuralNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test the model
model = ConvNeuralNet().to(device)
model_name = input("Enter model to load: ")
model.load_state_dict(torch.load('models/' + model_name))
print(f'"{model_name}" loaded.')

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {(100 * correct / total):.2f}%')