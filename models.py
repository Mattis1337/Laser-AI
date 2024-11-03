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
        from torch import nn
        x = torch.randn([0, 12, 8, 8])
        conv1 = nn.Conv2d(12, 24, 6)
        conv2 = nn.Conv2d(24, 48, 3)
        print(x.shape)
        x = conv1(x)
        print(x.shape)
        x = conv2(x)
        print(x.shape)
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


def init_neural_network(outputs: int, topology: list[nn.Sequential] = None):
    """
    Initialising a Neural Network using a specified topology / Sequential.
    :param outputs: number of outputs for the last layer
    :param topology: if provided the specified topology will be initialised
    """
    if topology is not None:
        return NeuralNetwork(topology[0], topology[1], outputs)

    topology = get_current_topology()

    return NeuralNetwork(topology[0], topology[1], outputs)


def get_current_topology():
    """
    Choosing one of the available network topologies using user input.
    """

    # Topologies need to be added manually, since hardcoding them
    # first of all is better to track all topologies that have been tested,
    # second of all makes it easier to have all systems on the same level
    # without having to use some weird method like pickling
    # and third of all is more secure since when just typing
    # a topology into a cli you are more prone to not notice mistakes
    available_topologies = {
        "Standard topology": STANDARD_TOPOLOGY,
        "Padded convolutional layer": PADDED_CONV_TOPOLOGY,
        "Upscaled fully connected layer": UPSCALED_FC_LAYERS,
        "Padded convolutional layer (without pooling)": PADDED_NOPOOL_TOPOLOGY,
    }
    invalid = True
    while invalid is True:
        print("Please choose one of the following topologies for training:")
        for i, topo in enumerate(available_topologies.keys()):
            print(i, f" {topo}")
            print(available_topologies[topo])
        index = input("Please insert the number of your option: ", )

        try:
            # getting the key by taking the wanted index of a list of all keys
            topology_key = [key for key in available_topologies.keys()][int(index)]
            return available_topologies[topology_key]
        except ValueError:
            print(f"Your option '{index}' is invalid!")


def get_output_shape(model, image_dim):
    """
    Returns torch.Size object to the last layer of a given model by providing it with dummy data and passing
    it through. The model will change the data and its dimension based on its different layers.
    :param model: the model topology
    :param image_dim: the dimensions of the dummy data
    """
    x = model(torch.rand(image_dim))
    return np.shape(x)


STANDARD_TOPOLOGY: list[nn.Sequential] = [
    nn.Sequential(
        nn.Conv2d(12, 24, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
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
PADDED_CONV_TOPOLOGY: list[nn.Sequential] = [
    nn.Sequential(
        nn.Conv2d(12, 24, 5, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
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

# Scaling up the fully connected layer will resolve overfitting and make the final output
# of the network more accurate
UPSCALED_FC_LAYERS: list[nn.Sequential] = [
    nn.Sequential(
        nn.Conv2d(12, 32, 5, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, 3),
        nn.ReLU(),
    ),
    nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
    )
]

# Removing the Pooling layer might lead to better results as pooling layers
# are generally being used to blurr out noise however on a chess board almost every feature
# plays a vital role to the move that should be generated
PADDED_NOPOOL_TOPOLOGY: list[nn.Sequential] = [
    nn.Sequential(
        nn.Conv2d(12, 32, 6, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 5),
        nn.ReLU(),
    ),
    nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
    )
]

