import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# Neural Network class
class RetroLaserAI(nn.Module):
    """
    For documentation measures the old Neural Network class will be left here.
    However, for purposes of benchmarking different topologies, nn.Sequentials()
    will be used to quickly define new models.
    """
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


class NeuralNetwork(nn.Module):
    def __init__(self, conv_seq, fc_seq, outputs):
        super().__init__()
        self.conv_seq = conv_seq
        self.fc_seq = fc_seq
        self.out_fc = nn.Linear(get_output_shape(fc_seq, get_output_shape(conv_seq, [12, 8, 8])[0])[0], outputs)

    def forward(self, x):
        x = self.conv_seq(x)
        x = torch.flatten(x, 1)
        x = self.fc_seq(x)
        x = self.out_fc(x)
        return x


def init_neural_network(outputs: int):
    """
    Initialising a Neural Network using a specified topology / Sequential.
    :param outputs: number of outputs for the last layer
    """
    conv_seq = STANDARD_TOPOLOGY[0]
    fc_seq = STANDARD_TOPOLOGY[1]

    model = NeuralNetwork(conv_seq, fc_seq, outputs)

    return model


STANDARD_TOPOLOGY: list[nn.Sequential()] = [
    nn.Sequential(
        nn.Conv2d(12, 24, 5),
        nn.MaxPool2d(2, 2),
        nn.ReLU(),
        nn.Conv2d(24, 32, 2),
        nn.ReLU(),
    ),
    nn.Sequential(
        nn.Linear(32, 120),
        nn.ReLU(),
        nn.Linear(120, 420),
        nn.ReLU(),
    )
]

# By adding zero_padding of 1 to the first conv layer the amount of kernel images
# fitting onto 1 layer gets larger giving us more weights to train
PADDED_CONV_TOPOLOGY: list[nn.Sequential()] = [
    nn.Sequential(
        nn.Conv2d(12, 24, 5, padding=1),
        nn.MaxPool2d(2, 2),
        nn.ReLU(),
        nn.Conv2d(24, 48, 3),
        nn.ReLU(),
    ),
    nn.Sequential(
        nn.Linear(48, 120),
        nn.ReLU(),
        nn.Linear(120, 420),
        nn.ReLU(),
    )
]


def get_output_shape(model, image_dim):
    """
    Returns torch.Size object to the last layer of a given model by providing it with dummy data and passing
    it through. The model will change the data and its dimension based on its different layers.
    :param model: the model topology
    :param image_dim: the dimensions of the dummy data
    """
    x = model(torch.rand(image_dim))
    return np.shape(x)
