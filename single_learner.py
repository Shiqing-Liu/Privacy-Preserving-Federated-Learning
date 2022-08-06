import torchvision
from torchvision import transforms
import os, time
from models import *
import matplotlib.pyplot as plt
import numpy as np

# M1 GPU support
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

TIME_STAMP = f"{time.localtime().tm_year}.{time.localtime().tm_mon}.{time.localtime().tm_mday}_{time.localtime().tm_hour}.{time.localtime().tm_min}"
SAVE_PATH = os.path.join(os.getcwd(), "results", "single_learner_" + TIME_STAMP)
os.mkdir(SAVE_PATH)

DATA_NAME = "CIFAR100"
EPOCHS = 10
BATCH_SIZE = 64
LR = 0.01
NUM_TRAIN_DATA = 1000
NUM_TEST_DATA = 1000
start = time.time()

# Get the correct model
if DATA_NAME == "MNIST":
    model = Net_2()
elif DATA_NAME == "CIFAR100":
    model = Net_4(100)
elif DATA_NAME == "CIFAR10":
    model = Net_4(10)
else:
    raise AssertionError("No fitting model found. Check your parameters!")
model = model.to(device)

# Get data
if DATA_NAME == "CIFAR10":  # 10 classes
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
elif DATA_NAME == "MNIST":  # 10 classes
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
elif DATA_NAME == "FashionMNIST":  # 10 classes
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    train_data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
elif DATA_NAME == "CIFAR100":  # 100 classes
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
else:
    raise NameError(f"No dataset named {DATA_NAME}. Choose from: CIFAR10, CIFAR100, MNIST, FashionMNIST")

indices = torch.randperm(len(train_data)).numpy()[:NUM_TRAIN_DATA]
train_data = torch.utils.data.Subset(train_data, indices)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

indices = torch.randperm(len(test_data)).numpy()[:NUM_TEST_DATA]
test_data = torch.utils.data.Subset(test_data, indices)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

# Training
optimizer = torch.optim.Adam
criterion = torch.nn.CrossEntropyLoss()
losses, accs = [], []

optimizer = optimizer(model.parameters(), lr=LR)
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}...", end="")
    model.train()
    for x, y in train_dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    print(f"\rEpoch {epoch + 1}/{EPOCHS} completed...", end="")
    model.eval()
    loss, acc = 0, 0
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss += criterion(outputs, y).item()
            preds = outputs.argmax(dim=1, keepdim=True)
            acc += (preds == y.view_as(preds)).sum().item()
    loss = loss / len(test_dataloader)
    acc = acc / len(test_data)

    losses.append(loss)
    accs.append(acc)
    print(f"\rEpoch {epoch + 1}/{EPOCHS} completed: loss: {loss}, accuracy: {acc}.")

with open(os.path.join(SAVE_PATH, "configuration.txt"), 'w') as f:
    f.write(f"The following training was conducted:\n\n")
    f.write(f"DATA_NAME: {DATA_NAME}\n")
    f.write(f"EPOCHS: {EPOCHS}\n")
    f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
    f.write(f"LR: {LR}\n")
    f.write(f"NUM_TRAIN_DATA: {NUM_TRAIN_DATA}\n")
    f.write(f"NUM_TEST_DATA: {NUM_TEST_DATA}\n")
    f.write(f"IID: {True}\n")
    dur = time.time() - start
    f.write(f"Duration: {int(dur//60)} minutes {round(dur%60)} seconds")

    f.write(f"\n\nResults: \n\n")
    f.write(f"Accs: {accs}\n")
    f.write(f"Losses: {losses}\n")

fig, ax = plt.subplots()
ax.plot(np.arange(1, EPOCHS+1, dtype="int32"), losses)
ax.set_xticks(np.arange(1, EPOCHS + 1, dtype="int32"))
ax.set_ylabel('Loss')
ax2 = ax.twinx()
ax2.plot(np.arange(1, EPOCHS+1, dtype="int32"), accs)
ax2.set_ylim([-0.05, 1.05])
plt.ylabel('Accuracy')


plt.title(f"Single Learner Performance")
ax.grid()
ax.legend(["Accuracy", "Loss"])
fig.savefig(os.path.join(SAVE_PATH, "performance.png"))



