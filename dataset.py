import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.ImageFolder("./Data/training_set/training_set/", transform)
test_dataset = torchvision.datasets.ImageFolder("./Data/test_set/test_set/", transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)