import chess
import numpy as np
import torch
from torch import nn
from torch import jit
import torch.nn.functional as F

import datasets

CURRENT_DEVICE = 'cpu'


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
    def __init__(self, conv_seq, fc_seq, fc_out, recurrent=False):
        super().__init__()
        self.conv_seq = conv_seq
        self.fc_seq = fc_seq
        self.recurrent = recurrent
        self.out_fc = fc_out

    def forward(self, x) -> torch.Tensor:
        x = self.conv_seq(x)
        x = torch.flatten(x, 1)
        if self.recurrent:
            x = torch.unsqueeze(x, 0)
        for module in self.fc_seq:
            if isinstance(module, nn.RNN):
                x = x.squeeze(-1)
                _, x = module(x)
                x = x[0]
            else:
                x = module(x)

        x = self.out_fc(x)
        return x


class LSTM(nn.Module):
    def __init__(self, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.layer_dim = layer_dim
        self.hidden_dim = 576
        self.conv1 = nn.Conv2d(12, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 4)
        # lstm1, lstm2, linear
        self.lstm = nn.LSTM(64, self.hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(576, 1152)
        self.out = nn.Linear(1152, output_dim)

    def forward(self, x, h0=None, c0=None):
        conv_out = []
        for i in range(x.size(0)):
            state = x[i, :, :, :]
            out = F.relu(self.conv1(state))
            out = F.relu(self.conv2(out))
            out = torch.flatten(out, 1)
            out = torch.squeeze(out, 1)
            conv_out.append(out)

        conv_out = torch.stack(conv_out)  # Shape: (batch_size, sequence_length, features)
        conv_out = torch.unsqueeze(conv_out, 0)
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, 1, self.hidden_dim, dtype=torch.float32).to(x.device)
            c0 = torch.zeros(self.layer_dim, 1, self.hidden_dim, dtype=torch.float32).to(x.device)

        # Forward pass
        out, (hn, cn) = self.lstm(conv_out, (h0, c0))
        # pass last hidden state for classification
        out = F.relu(self.fc(out))  # selecting the last output
        out = self.out(out)

        return out


def init_neural_network(outputs: int, topology=None):
    """
    Initialising a Neural Network using a specified topology / Sequential.
    :param outputs: number of outputs for the last layer
    :param topology: if provided the specified topology will be initialised
    """

    if isinstance(topology, LSTM):
        return LSTM(1, datasets.get_output_length(chess.BLACK))
    if topology is not None:
        fc_out = nn.Linear(get_output_shape(topology[1], get_output_shape(topology[0], [12, 8, 8])[0])[0], outputs)
        return NeuralNetwork(topology[0], topology[1], fc_out)

    topology = get_current_topology()

    if isinstance(topology, LSTM):
        return LSTM(2, datasets.get_output_length(chess.BLACK))

    fc_out = nn.Linear(get_output_shape(topology[1], get_output_shape(topology[0], [12, 8, 8])[0])[0], outputs)

    if isinstance(topology[-1], bool):
        return NeuralNetwork(topology[0], topology[1], fc_out, topology[-1])

    return NeuralNetwork(topology[0], topology[1], fc_out)


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
        "Unpadded convolutional layer (without pooling)": NOPOOL_BIGFC_LAYER,
        "Pooling Dropout Softmax": POOLING_DROPOUT,
        "Recurrent convolutional network": RECURRENT_CONV,
        "Recurrent convolutional network (smaller fc)": MINI_RECURRENT,
        "LSTM convolutional network": LSTM(1, 1),
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
    x = torch.rand(image_dim)

    # TODO: this has to be reconfigured (see todo1)
    for module in model:
        if isinstance(module, nn.RNN):
            x = torch.unsqueeze(x, 0)
            _, x = module(x)
        else:
            x = module(x)

    if isinstance(model[0], nn.RNN) or isinstance(model[0], nn.LSTM):
        return [np.shape(x)[1]]

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


NOPOOL_BIGFC_LAYER: list[nn.Sequential] = [
    nn.Sequential(
        nn.Conv2d(12, 32, 5),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4),
        nn.ReLU(),
    ),
    nn.Sequential(
        nn.Linear(64, 192),
        nn.ReLU(),
        nn.Linear(192, 576),
        nn.ReLU(),
        nn.Linear(576, 1728),
        nn.ReLU(),
    )
]

# testing rnns to bring more depth to the decision making of the AI

RECURRENT_CONV: list[nn.Sequential] = [
    nn.Sequential(
        nn.Conv2d(12, 32, 5),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4),
        nn.ReLU(),
    ),
    nn.Sequential(
        nn.RNN(64, 192, batch_first=True),
        nn.ReLU(),
        nn.Linear(192, 576),
        nn.ReLU(),
        nn.Linear(576, 1728),
        nn.ReLU(),
    ),
    True,
]

MINI_RECURRENT: list[nn.Sequential] = [
    nn.Sequential(
        nn.Conv2d(12, 32, 5),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4),
        nn.ReLU(),
    ),
    nn.Sequential(
        nn.RNN(64, 576, batch_first=True),
        nn.Tanh(),
        nn.Linear(576, 1152),
        nn.ReLU(),
    ),
    True,
]

LSTM_CONV: list[nn.Sequential] = [
    nn.Sequential(
        nn.Conv2d(12, 32, 5),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4),
        nn.ReLU(),
    ),
    nn.Sequential(
        nn.LSTM(64+64+64, 576, batch_first=True),
        nn.ReLU(),
        nn.Linear(576, 1152),
        nn.ReLU(),
    ),
    True,
]

POOLING_DROPOUT: list[nn.Sequential] = [
    nn.Sequential(
        nn.Conv2d(12, 32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3),
        nn.ReLU(),
    ),
    nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
    )
]
