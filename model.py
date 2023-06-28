import torch.nn as nn

class ConvNeuralNet(nn.Module):
    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        self.relu = nn.ReLU()                                                               # input:    128, 128, 3
        self.conv1 = nn.Conv2d(3, 10, kernel_size=6, padding=2, stride=2)                   #           64, 64, 10
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                                  #           32, 32, 10
        self.conv2 = nn.Conv2d(10, 18, kernel_size=4, padding=2, stride=2)                  #           17, 17, 18
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)                                  #           8, 8, 18
        self.fc1 = nn.Linear(in_features=(8*8*18), out_features=128)                        
        self.fc2 = nn.Linear(128, 2)
    def forward(self, inputs):
        # 1st Conv layer
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.pool1(x)
        # 2nd Conv layer
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully Connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    