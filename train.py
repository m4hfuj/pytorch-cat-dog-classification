import os
import time
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from dataset import train_loader
from model import ConvNeuralNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters
learning_rate = 0.003
num_epochs = 100
# learning_rate = float(input("Enter Learning rate for the model: "))
# num_epochs = int(input("Enter number of Epoch: "))
model_name = input("Enter model name to save: ")

""" For Time """
def date(secs):
    if secs < 60.0:
        return "0." + str( int(secs) )
    else:
        m = str( int(secs / 60) )
        s = str( int(secs % 60) )
        return m + "." + s
start = time.time()
et = 0.0

# Model and Others
model = ConvNeuralNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training
total_step = len(train_loader)
print(f'No. of total steps: {total_step}')
print("starting training...")
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            os.system('cls')
            print(f'Learning rate     : {learning_rate}')
            print(f'Number of Epochs  : {num_epochs}')
            print(f'No of total steps : {total_step}')

            print(f'\nTraining going on....\n')

            et = time.time() - start
            print(f'Elapsed time      : {date(et)} s')
            print(f'\nEpoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

# Save the model
print("stoping training...")
torch.save(model.state_dict(), 'models/' + model_name)
print(f'model saved as "{model_name}".')
print("training stopped.")
