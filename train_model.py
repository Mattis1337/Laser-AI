import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import datasets
import atexit


# TODO: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# Neural Network class
class NeuralNetwork(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        # NOTE: these parameters are still up for optimization
        self.conv1 = nn.Conv2d(batch_size, 6, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 2)
        self.fc1 = nn.Linear(1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Single iteration training
def train(dataloader, model, criterion, optimizer):
    """
    Training a chess model depending on the color
    :param dataloader: the dataloader containing the white/black moves which shall be used for training
    :param model: a model object instantiated from NeuralNetwork
    :param criterion: loss function
    :param optimizer: optimizer
    """

    size = len(dataloader.dataset)

    for batch, data in enumerate(dataloader):
        inputs, labels = data  # not adjusted for CUDA devices
        print(inputs)

        # zero the parameter gradient
        optimizer.zero_grad()

        # forward + backward + optimize
        pred = model(inputs)
        loss = criterion(pred, torch.from_numpy(labels))  # TODO: turn all possible outputs into usable tensor type
        loss.backward()
        optimizer.step()

        # statistics ...
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(inputs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    print("Epoch done!")


def test(dataloader, device, model, criterion,):
    """
    Testing the accuracy of a given model
    :param dataloader: the dataloader containing the white/black moves which shall be used for testing
    :param device: the device (e.g. cpu, cuda) which shall be used for testing
    :param model: a model object instantiated from NeuralNetwork
    """

    for X, y in dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def exit_handler(model, save_file):
    """
    Saving the state of the model when process is manually stopped so that you won't have to wait out
    the whole training process before train_module finishes.
    :param model: the model whose state shall be saved
    :param save_file: the file in which the model state shall be saved
    """

    torch.save(model.state_dict(), save_file)
    print(f"Saved PyTorch Current Model State to {save_file}")


# full iterations training
def train_chess_model(dataset: datasets.ChessDataset) -> None:
    """
    This function will train a neural network by creating an instance of
    the neural network class loading the according weights onto it and then
    using the given dataset to train.
    :param dataset: instance of a custom dataset class
    """

    # batch size (adjust if training is too slow or the hardware is not good enough)
    batch_size = dataset.__len__()  # Use datasets len() func to get the length of the whole data

    train_dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4)

    # Initialising the module
    model = NeuralNetwork(batch_size)
    print(model)

    # registering the exit handler with the right path
    # the enum COLOR of the chess library is true when it is white
    if dataset.__color__() is True:
        atexit.register(exit_handler, model, "white_model.pth")

    elif dataset.__color__() is False:
        atexit.register(exit_handler, model, "black_model.pth")

    else:
        raise ValueError("Dataset must be either of the instance of white or black but neither was given.")

    # Setting the module parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    # Train the network for the set epoch size
    epochs = 5
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, criterion, optimizer)
        test(test_dataloader, model, criterion, optimizer)

    print("Done!")

    # Saving the models state to the correct file
    if dataset.__color__:
        exit_handler(model, "white_model.pth")
    elif not dataset.__color__:
        exit_handler(model, "black_model.pth")


# TODO: generate a move to a given position by initialising a model with the according weights
def generate_move(color, game_state):
    """
    When using the AI this function will return the move for a given
    game state.
    :param color: what type of AI is to be trained
    :param game_state: the state of the game which a move is to be generated for
    """
    # Get cpu, gpu or mps device for training.
    hardware_device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {hardware_device} device")

    # Initialising a model
    model = NeuralNetwork(1).to(hardware_device)

    # Loading the correct state to the model
    if color == "white":
        model.load_state_dict(torch.load("model_white.pth"))

    elif color == "black":
        model.load_state_dict(torch.load("model_black.pth"))

    classes = []  # Saves all possible outputs for the nn

    path = 'moves.csv'  # Personal path to the file with all moves for a color
    # TODO: depending on the color of the dataset load the moves csv of that color

    model.eval()
    x = game_state  # Game state must be the same data type the network trains with

    with torch.no_grad():
        x = x.to(hardware_device)
        pred = model(x)
        predicted = classes[pred[0].argmax(0)]
        print(f'Predicted: "{predicted}"')
