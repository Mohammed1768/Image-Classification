import matplotlib.pyplot as plt
from NeuralNet import NeuralNet 
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

model = NeuralNet()
model.load_state_dict(torch.load("NN\\fashion_model.pth", map_location="cpu"))
model.eval()

labels = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}

testing_set = datasets.FashionMNIST(
    root="data",
    train=1,
    download=1,
    transform=ToTensor()
)

k = torch.randint(low=0, high=10000, size=(1,)).item()
for num in range(12):
    img,label = testing_set[k + num]

    plt.subplot(2,6, 1+num)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(f"Prediction={labels[model.forward(img.unsqueeze(0)).argmax(1).item()]}\n Actual={labels[label]}")
    plt.axis(False)
plt.show()