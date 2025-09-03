import torch
from NeuralNet import NeuralNet
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import SGD
import matplotlib.pyplot as plt


training_set = FashionMNIST(
    root="data",
    train=1,
    download=1,
    transform=ToTensor()
)

map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

learning_rate = 0.01
epochs = 15
batch_size = 50


model = NeuralNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = SGD(params=model.parameters(), lr=learning_rate)

data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=1)

for epoch in range(1,1+epochs):
    error = 0
    for x,y in data_loader:
        out = model.forward(x)
        loss = loss_fn(out, y)

        error += loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    print(f"Epoch:{epoch}, Loss: {error/len(data_loader)}")
    
torch.save(model.state_dict(), "fashion_model.pth")
