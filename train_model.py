import os.path

import chess
import chess as c
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# importing own files
import chess_annotation
import datasets
import data_transformations as dt


OUTPUTS_WHITE: dict[str, int] = dt.targets_to_tensor(c.WHITE)
OUTPUTS_BLACK: dict[str, int] = dt.targets_to_tensor(c.BLACK)


# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# Neural Network class
class NeuralNetwork(nn.Module):
    def __init__(self, outputs):
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
        self.fc1 = nn.Linear(32, 120)  # this seems kinds sketchy...
        self.fc2 = nn.Linear(120, 420)
        self.fc3 = nn.Linear(420, outputs)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def initialize_model(color: chess.Color):
    """
    Creating a new model state by initializing an untrained model and
    saving it to a desired path.
    :param color: the type of model created
    """
    # getting the outputs matching the color for the topology
    outputs = datasets.get_output_length(color)
    # setting the fitting topology of an untrained network
    model = NeuralNetwork(outputs)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    # TODO: rework saving and loading of states to fit multiple states of
    #  the same color type and make initialization more secure

    # entering new path to save the model to
    path = input('File to save new model state: ')

    # saving untrained model with unused optimizer
    save_trained_model(color, model, 0, optimizer, path)


# Single iteration training
def train(dataloader, model, criterion, optimizer, device):
    """
    Training a chess model depending on the color
    :param dataloader: the dataloader containing the white/black moves which shall be used for training
    :param model: a model object instantiated from NeuralNetwork
    :param criterion: loss function
    :param optimizer: optimizer
    :param device: the device currently used for training
    """

    size = len(dataloader.dataset)
    running_loss = 0.0

    for batch, data in enumerate(dataloader, 0):
        inputs, labels = data  # not adjusted for CUDA devices

        # zero the parameter gradient
        optimizer.zero_grad(set_to_none=True)

        # forward + backward + optimize
        pred = model(inputs.to(device))
        loss = criterion(pred, labels.to(device))
        loss.backward()
        optimizer.step()

        # statistics ...
        running_loss += loss
        if (batch+1) % 1000 == 0:
            current = (batch + 1) * len(inputs)
            print(f"loss: {running_loss / (dataloader.batch_size * 1000):>7f}  [{current:>5d}/{size:>5d}]")
            running_loss = 0.0

    print("Epoch done!")


def test(dataloader, model, device):
    """
    Testing the accuracy of a given model by calculating the total error in a given output by averaging the error per
    digit in an output and adding it.
    :param dataloader: the dataloader containing the white/black moves which shall be used for testing
    :param model: a model object instantiated from NeuralNetwork
    :param device: the device currently used for training
    """

    total = 0
    # possible_targets = datasets.get_output_length(dataloader.dataset.__color__())
    color = dataloader.dataset.__color__()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            image, label = data
            targets = dataloader.dataset.__gettargets__()

            # calculate outputs by running game states through the network
            pred = model(image.to(device))

            pred = dt.tensor_to_targets(pred, targets, amount_targets=1).to(device)

            if dt.compare_tensors(label[0].to(device), pred) is True:
                total += 1

            if (i+1) % 10000 == 0:
                print(f'Successfully tested {i+1} randomly shuffled game states!')
                break

    print(f'Accuracy of the network: {total/100}%')


# full iterations training
def train_chess_model(dataset: datasets.ChessDataset, epochs: int) -> None:
    """
    This function will train a neural network by creating an instance of
    the neural network class loading the according weights onto it and then
    using the given dataset to train.
    :param dataset: instance of a custom dataset class
    :param epochs: number of epochs of training
    """

    # Reference for handling changing dimensions
    # https://discuss.pytorch.org/t/how-to-load-a-dpretrained-model-with-a-different-output-dimension/26117/7

    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(dataset, shuffle=True, num_workers=8)

    # loading the last checkpoints
    state, path = load_model()
    # casting the number of old outputs
    old_outputs = state['output_size']

    # getting the device which should be used for training
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Currently using {device} device!")

    # loading the model
    model = NeuralNetwork(old_outputs).to(device)

    # Setting the module parameters
    criterion = nn.CrossEntropyLoss()
    # https://amarsaini.github.io/Optimizer-Benchmarks/
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    # checking if the output size has changed in between learning
    if old_outputs != datasets.get_output_length(dataset.__color__()):
        # change the dimension of the output
        num_ftrs = model.fc3.in_features
        model.fc3 = nn.Linear(num_ftrs, old_outputs)
        # loading the pre-trained model with the old outputs
        model.load_state_dict(state['model_state_dict'], strict=False)
        # changing the model to new output size
        model.fc3 = nn.Linear(num_ftrs, datasets.get_output_length(dataset.__color__()))
    else:
        # loading optimizer and model state only when the output size has not changed
        optimizer.load_state_dict(state['optimizer_state_dict'])
        model.load_state_dict(state['model_state_dict'])

    # setting model into training mode
    model.train()

    last_epoch = state['epoch']

    # Train the network for the set epoch size
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, criterion, optimizer, device)

        if (epoch+1) % 20 == 0:
            # testing and saving the model after 20 epochs
            test(test_dataloader, model, device)
            save_trained_model(dataset.__color__(), model, last_epoch + 20, optimizer, path)

    # testing the model after all epochs
    test(test_dataloader, model, device)

    print("Done!")

    # saving the model after it finished training
    save_trained_model(dataset.__color__(), model, last_epoch + epochs, optimizer, path)


def generate_move(color, fen, amount_outputs=1):
    """
    When using the AI this function will return the move for a given
    game state.
    :param color: what type of AI is to be trained
    :param fen: current game state
    :param amount_outputs: the top amount of targets to be returned
    """

    # loading the model
    state, path = load_model()

    model = NeuralNetwork(datasets.get_output_length(color))
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    # turning the fen into bitboards
    bbs = chess_annotation.fen_to_bitboards(fen)
    bbs = np.array(bbs)
    # bitboards need to be cast to an array for compatibility
    x = dt.transform_bitboards(bbs)
    # adjusting the dimension of the input to match with batch (took me 4 hours to fix...)
    x_ = np.empty([1, 12, 8, 8], dtype=np.float32)
    x_[0] = x
    # turning the array into tensor
    x = dt.to_tensor(x_)

    outputs: dict[str, int]
    match color:
        case c.WHITE:
            outputs = OUTPUTS_WHITE
        case c.BLACK:
            outputs = OUTPUTS_BLACK
        case _:
            raise ValueError("Expected variable color to be of type bool but received" + type(color))

    with torch.no_grad():
        pred = model(x)
        pred = dt.tensor_to_targets(pred,
                                    outputs,
                                    amount_outputs)

        print(f'Predicted: "{pred}"')

    return pred


def load_model():
    """
    Initialise a NeuralNetwork model using a .pth file to load
    the weights to the specified color of that model.
    """

    # TODO: adapt saving and loading system to different path names
    path = input('Enter path to model: ')

    state = torch.load(path)

    return state, path


def save_trained_model(color, model, epoch, optimizer, path):
    """
    Saving a NeuralNetwork model after training in a specified file
    based off the color.
    :param color: specifies which save file should be used
    :param model: the trained model which should be saved
    :param epoch: what epoch training was left on
    :param optimizer:  the  adjusted optimizer which was used
    :param path: path to save the state to
    """
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'output_size': datasets.get_output_length(color)},
               path)
    print(f"Saved PyTorch Current Model State to {path}")
