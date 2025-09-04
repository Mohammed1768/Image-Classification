from CNN import CNN
from torchvision.transforms import ToTensor
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import torch

training_set = FashionMNIST(
    root="..\\data",
    train=1,
    download=1,
    transform=ToTensor()
)

model = CNN()

learning_rate = 0.005

epochs = 10
batch_size = 64

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

dataloader = DataLoader(training_set, batch_size=batch_size)

for epoch in range(1, 1+epochs):
    error = 0
    for (x,y) in dataloader:
        pred = model(x)   
        loss = loss_fn(pred, y)   

        optimizer.zero_grad()      
        loss.backward()              
        optimizer.step()     

        error += loss
    scheduler.step()
    print(f"Epoch:{epoch}, Loss: {error/len(dataloader)}")       


torch.save(model.state_dict(), "fashion_model.pth")
print("Saved")