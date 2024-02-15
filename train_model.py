import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import annotation_converter
from datasets import WhiteMovesDataset, BlackMovesDataset
import atexit


# INITIALISING THE DATA

# Initialising the data

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


# Neural Network class
class NeuralNetwork(nn.Module):
    """The Neural Network class"""
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


def train(dataloader, device, model, loss_fn, optimizer):
    """
    Training a chess model depending on the color
    :param dataloader: the dataloader containing the white/black moves which shall be used for training
    :param device: the device (e.g. cpu, cuda) which shall be used for training
    :param model: a model object instantiated from NeuralNetwork
    :param loss_fn: loss function
    :param optimizer: optimizer
    """

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


def test(dataloader, device, model, loss_fn,):
    """
    Testing the accuracy of a given model
    :param dataloader: the dataloader containing the white/black moves which shall be used for testing
    :param device: the device (e.g. cpu, cuda) which shall be used for testing
    :param model: a model object instantiated from NeuralNetwork
    :param loss_fn: loss function
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
            test_loss += loss_fn(pred, y).item()
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


def create_chess_model(dataset):
    """
    This function will train a neural network by creating an instance of
    the neural network class loading the according weights onto it and then
    using the given dataset to train.
    :param dataset: instance of a custom dataset class
    """

    # batch size (adjust if training is too slow or the hardware is not good enough)
    batch_size = dataset.__len__()  # Use datasets len() func to get the length of the whole data

    train_dataloader = DataLoader(dataset, batch_size)
    test_dataloader = DataLoader(dataset, batch_size)

    # Get cpu, gpu or mps device for training.
    hardware_device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {hardware_device} device")

    # Initialising the module
    model = NeuralNetwork().to(hardware_device)
    print(model)

    # registering the exit handler with the right path
    if isinstance(dataset, WhiteMovesDataset):
        atexit.register(exit_handler, model, "white_model.pth")

    elif isinstance(dataset, BlackMovesDataset):
        atexit.register(exit_handler, model, "black_model.pth")

    else:
        raise ValueError("Dataset must be either of the instance of white or black but neither was given.")

    # Setting the module parameters
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Train the network for the set epoch size
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, hardware_device, model, loss_fn, optimizer)
        test(test_dataloader, hardware_device, model, loss_fn)

    print("Done!")

    # Saving the models state to the correct file
    if isinstance(dataset, WhiteMovesDataset):
        torch.save(model.state_dict(), "model_white.pth")
        print("Saved PyTorch Model State to model_white.pth")
    else:
        torch.save(model.state_dict(), "model_white.pth")
        print("Saved PyTorch Model State to model_white.pth")


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
    model = NeuralNetwork().to(hardware_device)

    # Loading the correct state to the model
    if color == "white":
        model.load_state_dict(torch.load("model_white.pth"))

    elif color == "black":
        model.load_state_dict(torch.load("model_black.pth"))

    classes = []  # Saves all possible outputs for the nn

    path = '/home/mattis/Documents/Jugend_Forscht_2023.24/all_moves.csv'  # Personal path to the file with all moves

    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            classes.append(annotation_converter.move_filter(line))

    model.eval()
    x = game_state  # Game state must be the same data type the network trains with

    with torch.no_grad():
        x = x.to(hardware_device)
        pred = model(x)
        predicted = classes[pred[0].argmax(0)]
        print(f'Predicted: "{predicted}"')
