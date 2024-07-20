import chess as c
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import datasets


# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# Neural Network class
class NeuralNetwork(nn.Module):
    def __init__(self):
        """
        The constructor of the neural network determines the topology of the network.
        The parameters in channels are always fixed based off the number of output channels derived
        from the layer before.

        The following script calculates the number of input channels for fc1.

        # calculate the input channel for fully connected layer 1
        x, y = dataset.__getitem__(0)
        conv1 = nn.Conv2d(12, 24, 5)
        pool = nn.MaxPool2d(2, 2)
        conv2 = nn.Conv2d(24, 32, 2)
        x = conv1(x)
        x = pool(x)
        x = conv2(x)
        print(x.shape)
        # based of the dimension of 32x1x1 one can deduce that only 1 pooling layer is needed
        """
        super().__init__()
        # 12 input channels for 12 different piece types
        self.conv1 = nn.Conv2d(12, 24, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(24, 32, 2)
        self.fc1 = nn.Linear(32*1*1, 120)  # this seems kinds sketchy...
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
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
    running_loss = 0.0

    for batch, data in enumerate(dataloader, 0):
        inputs, labels = data  # not adjusted for CUDA devices

        # zero the parameter gradient
        optimizer.zero_grad()

        # forward + backward + optimize
        pred = model(inputs)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

        # statistics ...
        running_loss += loss
        if (batch+1) % 2000 == 0:
            current = (batch + 1) * len(inputs)
            print(f"loss: {running_loss / (dataloader.batch_size * 2000):>7f}  [{current:>5d}/{size:>5d}]")
            running_loss = 0.0

    print("Epoch done!")


def test(dataloader, model):
    """
    Testing the accuracy of a given model by calculating the total error in a given output by averaging the error per
    digit in an output and adding it.
    :param dataloader: the dataloader containing the white/black moves which shall be used for testing
    :param model: a model object instantiated from NeuralNetwork
    """

    total = 0.0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            # calculate outputs by running game states through the network
            outputs = model(images)
            t_out_error = 0.0
            for i in range(8):
                out = float(outputs[0][i])
                tar = float(labels[0][i])
                t_out_error += abs(out - tar)

            total += (t_out_error/8)

        total = total / (dataloader.batch_size * dataloader.__len__())

    print(f'Inaccuracy of the network: {total}')


# full iterations training
def train_chess_model(dataset: datasets.ChessDataset, epochs: int) -> None:
    """
    This function will train a neural network by creating an instance of
    the neural network class loading the according weights onto it and then
    using the given dataset to train.
    :param dataset: instance of a custom dataset class
    :param epochs: number of epochs of training
    """

    # batch size (adjust if training is too slow or the hardware is not good enough)
    # batch_size = dataset.__len__()  # Use datasets len() func to get the length of the whole data

    train_dataloader = DataLoader(dataset, 20, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(dataset, shuffle=True, num_workers=4)

    # loading the last checkpoint
    state = load_model(dataset.__color__())

    # loading the model
    model = NeuralNetwork()
    model.load_state_dict(state['model_state_dict'])
    # setting model into training mode
    model.train()

    # Setting the module parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    optimizer.load_state_dict(state['optimizer_state_dict'])

    last_epoch = state['epoch']

    # Train the network for the set epoch size
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, criterion, optimizer)

        if (epoch+1) % 20 == 0:
            # testing the model after 10 epochs
            test(test_dataloader, model)
            # saving model after 20 epochs
            save_trained_model(dataset.__color__(), model, last_epoch + 20, optimizer)

    # testing the model after all epochs
    test(test_dataloader, model)

    print("Done!")

    # saving the model after it finished training
    save_trained_model(dataset.__color__(), model, last_epoch+epochs, optimizer)


# TODO: interpret the output given by the model
def generate_move(color, dataset: datasets.ChessTestData):
    """
    When using the AI this function will return the move for a given
    game state.
    :param color: what type of AI is to be trained
    :param dataset: a test dataset containing only 1 game state
    """

    # loading the model
    state = load_model(color)

    model = NeuralNetwork()
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    loader = DataLoader(dataset, batch_size=1)

    with torch.no_grad():
        for data in loader:
            x, y = data
            pred = model(x)
            print(f'Predicted: "{pred}"')


def load_model(color):
    """
    Initialise a NeuralNetwork model using a .pth file to load
    the weights to the specified color of that model.
    :param color: specifies what weights should be loaded
    """

    if color is True:
        state = torch.load('white_model.pth')

    elif color is False:
        state = torch.load('black_model.pth')

    else:
        raise ValueError(f"Dataset must be either of the instance of {c.WHITE} or {c.BLACK} "
                         f"but {color} was given.")

    return state


def save_trained_model(color, model, epoch, optimizer):
    """
    Saving a NeuralNetwork model after training in a specified file
    based off the color.
    :param color: specifies which save file should be used
    :param model: the trained model which should be saved
    :param epoch: what epoch training was left on
    :param optimizer:  the  adjusted optimizer which was used
    """
    if color is True:
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   'white_model.pth')
        print(f"Saved PyTorch Current Model State to white_model.pth")
    elif color is False:
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   'black_model.pth')
        print(f"Saved PyTorch Current Model State to black_model.pth")
