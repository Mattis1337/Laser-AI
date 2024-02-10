import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from annotation_converter import create_input_datasets
from data_preparer import WhiteMovesDataset, BlackMovesDataset
import atexit

# TODO: If everything starts working maybe refactor the way this file is structured 
#  meaning to create functions for training either the black moves neural network or the white moves neural network

#####INITIALISING THE DATA#####

# Initialising the data
create_input_datasets() 

path_white_img = '/home/mattis/Documents/Jugend_Forscht_2023.24/finished_data/white_img'
path_white_labels = '/home/mattis/Documents/Jugend_Forscht_2023.24/finished_data/white_labels'
path_black_img = '/home/mattis/Documents/Jugend_Forscht_2023.24/finished_data/black_img'
path_black_labels = '/home/mattis/Documents/Jugend_Forscht_2023.24/finished_data/black_labels'

# Getting training data:
training_data_white = WhiteMovesDataset(path_white_labels, path_white_img, ToTensor)
training_data_black = BlackMovesDataset(path_black_labels, path_black_img, ToTensor)

# Getting data for testing
test_data_white = WhiteMovesDataset(path_white_labels, path_white_img, ToTensor)
test_data_black = BlackMovesDataset(path_black_labels, path_black_img, ToTensor)


# TODO: DataLoaders are created before training -> Batch size needs to be a big number and if there is no next move
#  then break

def batch_length(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        length = 0
        print(lines)
        for num in lines:
            print(num)
            length+=1
    
    print(length)
    return length

batch_size_white = batch_length(path_white_img)
batch_size_black = batch_length(path_black_img)

# Create data loaders white
train_dataloader_white = DataLoader(training_data_white, batch_size=batch_size_white)
test_dataloader_white = DataLoader(test_data_white, batch_size=batch_size_white)
# Create data loaders black
train_dataloader_black = DataLoader(training_data_black, batch_size=batch_size_black)
test_dataloader_black = DataLoader(test_data_black, batch_size=batch_size_black)

# TODO: Maybe implement maybe get rid of this
#for X, y in test_dataloader:
#   print(f"Shape of X [N, C, H, W]: {X.shape}")
#    print(f"Shape of y: {y.shape} {y.dtype}")
#    break

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")



# Neural Network class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model_white = NeuralNetwork().to(device)
print(model_white)

model_black = NeuralNetwork().to(device)
print(model_black)

loss_fn = nn.CrossEntropyLoss()
optimizer_white = torch.optim.SGD(model_white.parameters(), lr=1e-3)
optimizer_black = torch.optim.SGD(model_black.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# TODO: Adjusting the epochs to the wanted amount or basically infinity
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    # White
    train(train_dataloader_white, model_white, loss_fn, optimizer_white)
    test(test_dataloader_white, model_white, loss_fn)
    # Black
    train(train_dataloader_black, model_black, loss_fn, optimizer_black)
    test(test_dataloader_black, model_black, loss_fn)
print("Done!")

#####SAVING MODELS#####

torch.save(model_white.state_dict(), "model_white.pth")
print("Saved PyTorch White Model State to model_white.pth")

torch.save(model_white.state_dict(), "model_white.pth")
print("Saved PyTorch White Model State to model_white.pth")

#####LOADING MODELS#####

model_white = NeuralNetwork().to(device)
model_white.load_state_dict(torch.load("model_white.pth"))

model_black = NeuralNetwork().to(device)
model_black.load_state_dict(torch.load("model_black.pth"))


classes = []  # Saves all possible outputs for the nn

path = '/home/mattis/Documents/Jugend_Forscht_2023.24/all_moves.csv'  # Personal path to the file with all moves

with open(path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        classes.append(annotation_converter.move_filter(line))

model_white.eval()
x, y = test_data_white[0][0], test_data_white[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model_white(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

model_black.eval()
x, y = test_data_black[0][0], test_data_black[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model_black(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

# TODO: Figure out file where the state is saved
def exit_handler():
    torch.save(model_white.state_dict(), "model_white.pth")
    print("Saved PyTorch White Model State to model.pth")
    torch.save(model_black.state_dict(), "model_black.pth")
    print("Saved PyTorch White Model State to model.pth")


atexit.register(exit_handler)
