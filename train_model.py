import os.path

import chess
import chess as c
import numpy as np
import torch
from numpy.f2py.auxfuncs import throw_error
from torch import nn
from torch.utils.data import DataLoader

# importing own files
import chess_annotation
import datasets
import data_transformations as dt
import models
from data_transformations import targets_to_numericals

OUTPUTS_WHITE: dict[str, int] = dt.targets_to_numericals(c.WHITE)
OUTPUTS_BLACK: dict[str, int] = dt.targets_to_numericals(c.BLACK)


def init_new_model():
    """
    Creating a new model state by initializing an untrained model and
    saving it to a desired path.
    """
    # getting the wanted colour
    while True:
        print('Following color options: ' + '\n' +
              '1) Black' + '\n' +
              '2) White')

        col_index = int(input('Pick an option 1-2:'))
        if col_index in [1, 2]:
            break
        print(f'Specified option {col_index} is invalid!')

    if col_index == 1:
        color = chess.BLACK
    else:
        color = chess.WHITE
    # getting the outputs matching the color for the topology
    outputs = datasets.get_output_length(color)
    # setting the fitting topology of an untrained network
    model = models.init_neural_network(outputs)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    # entering new path to save the model to
    path = input('Insert name for the new model state (.pth will be appended!): ')

    # saving untrained model with unused optimizer
    save_trained_model(color, model, 0, optimizer, path+'.pth')


# Single iteration training
def train_cnn(dataloader, model, criterion, optimizer, device):
    """
    Training a feed forward chess model
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
        pred, _ = model(inputs.to(device))
        loss = criterion(pred, labels.to(device))
        loss.backward()
        optimizer.step()

        # statistics ...
        running_loss += loss
        if (batch+1) % 10000 == 0:
            current = (batch + 1) * len(inputs)
            print(f"loss: {running_loss / (dataloader.batch_size * 10000):>7f}  [{current:>5d}/{size:>5d}]")
            running_loss = 0.0

    print("Epoch done!")


def train_rnn(dataloader, model, criterion, optimizer, device):
    """
    Training a recurrent neural network
    :param dataloader: the dataloader containing the white/black moves which shall be used for testing
    :param model: a model object instantiated from NeuralNetwork
    :param criterion: loss function
    :param optimizer: the optimizer used for enhancing the training algorithm
    :param device: the device currently used for training
    """
    running_loss = 0
    for batch, data in enumerate(dataloader, 0):
        input_sequence, target_sequence = data
        # TODO: when fitting dataset is available the data dimensions will change
        hidden = model.initHidden()

        model.zero_grad()
        optimizer.zero_grad(set_to_none=True)

        loss = torch.Tensor([0])  # you can also just simply use ``loss = 0``

        # getting rid of the batch dimension which effectively is 1 all the time
        input_sequence = input_sequence[0]
        target_sequence = target_sequence[0]

        # iterating through all previous moves
        for i in range(input_sequence.size(0)):
            output, hidden = model(input_sequence[i].to(device), hidden.to(device))
            target = torch.unsqueeze(target_sequence[i], dim=0)
            l = criterion(output.to(device), target.to(device))
            loss += l

        if loss.requires_grad is not False:
            loss.backward()
            optimizer.step()
        else:
            print("passing")
        running_loss += loss

        if batch % 100 == 0:
            print(f"loss: {running_loss / batch}  [{batch:>5d}/{len(dataloader.dataset):>5d}]")
            running_loss = 0


def test_cnn(dataloader, model, device):
    """
    Testing the accuracy of a given model by calculating the total error in a given output by averaging the error per
    digit in an output and adding it.
    :param dataloader: the dataloader containing the white/black moves which shall be used for testing
    :param model: a model object instantiated from NeuralNetwork
    :param device: the device currently used for testing
    """

    model.eval()

    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            image, label = data

            # calculate outputs by running game states through the network
            pred, _ = model(image.to(device))
            # getting the highest match of the AI output
            pred = dt.get_highest_index(pred[0], 1)

            # comparing if the AI's match is the same as the output
            if int(label[0]) == int(pred[0]):
                total += 1

            if (i+1) % 10000 == 0:
                print(f'Successfully tested {i+1} randomly shuffled game states!')
                break

    print(f'Accuracy of the network: {total/100}%')
    model.train()


def test_rnn(dataloader, model, device):
    """
    Testing the accuracy of a given recurrent model by calculating the total error in a given output by averaging the error per
    digit in an output and adding it.
    :param dataloader: the dataloader containing the white/black moves which shall be used for testing
    :param model: a model object instantiated from NeuralNetwork
    :param device: the device currently used for testing
    """

    model.eval()

    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            input_sequence, label = data

            if model.recurrent is True:
                hidden = model.initHidden()
            # calculate outputs by running game states through the network
            for j in range(input_sequence.size(0)):
                output, hidden = model(input_sequence[j].to(device), hidden.to(device))

                # getting the highest match of the AI output
                pred = dt.get_highest_index(output[0], 1)

                # comparing if the AI's match is the same as the output
                if int(label[0]) == int(pred[0]):
                    total += 1

            if (i+1) % 10000 == 0:
                print(f'Successfully tested {i+1} randomly shuffled game states!')
                break

    print(f'Accuracy of the network: {total/100}%')
    model.train()


# full iterations training
def train_chess_model() -> None:
    """
    This function will train a neural network by creating an instance of
    the neural network class loading the according weights onto it and then
    using the given dataset to train.
    """
    # Getting the number of epochs
    while True:
        try:
            epochs = int(input('Set number of epochs of learning: '))
            break
        except ValueError:
            print(f'Expected epochs to be of type {int}!')

    # disabling debugging APIs
    set_debug_apis(False)

    # loading the last checkpoints
    state, path = load_model()

    # getting the device which should be used for training
    device = get_training_device()

    # casting the number of old outputs and the models color
    old_outputs = state['output_size']
    color = state['color']

    # Initializing NeuralNetwork
    model = models.init_neural_network(old_outputs).to(device)
    # Setting the module parameters
    criterion = nn.CrossEntropyLoss()
    # https://amarsaini.github.io/Optimizer-Benchmarks/
    learning_rate = 1e-4
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # checking if the output size has changed in between learning
    if old_outputs != datasets.get_output_length(color):
        # change the dimension of the output
        if isinstance(model, models.NeuralNetwork):
            num_ftrs = model.out_fc.in_features
            output_layer = model.out_fc
            model.out_fc = nn.Linear(num_ftrs, old_outputs)
            # loading the pre-trained model with the old outputs
            model.load_state_dict(state['model_state_dict'], strict=False)
            # changing the model to new output size
            model.out_fc = nn.Linear(num_ftrs, datasets.get_output_length(color))
        else:
            num_ftrs = model.fc3.in_features
            output_layer = model.fc3
            model.fc3 = nn.Linear(num_ftrs, old_outputs)
            # loading the pre-trained model with the old outputs
            model.load_state_dict(state['model_state_dict'], strict=False)
            # changing the model to new output size
            model.fc3 = nn.Linear(num_ftrs, datasets.get_output_length(color))
    else:
        # loading optimizer and model state only when the output size has not changed
        optimizer.load_state_dict(state['optimizer_state_dict'])
        model.load_state_dict(state['model_state_dict'])

    # setting model into training mode
    model.train()
    last_epoch = state['epoch']

    # initializing the dataset
    if model.recurrent is True:
        dataset = datasets.init_chess_dataset(color, True)
        print("work")
    else:
        dataset = datasets.init_chess_dataset(color, False)

    # Initializing Dataloaders
    if model.recurrent is True:
        train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)
        test_dataloader = DataLoader(dataset, shuffle=False, num_workers=8)
    else:
        train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=16)
        test_dataloader = DataLoader(dataset, shuffle=False, num_workers=8)

    # Printing info
    print(f'Resuming training at epoch {last_epoch}!')
    # Number of trainable parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Train the network for the set epoch size
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} (total {last_epoch+epoch+1})\n-------------------------------")
        if model.recurrent is True:
            train_rnn(train_dataloader, model, criterion, optimizer, device)
        else:
            train_cnn(train_dataloader, model, criterion, optimizer, device)

        if (epoch+1) % 1 == 0:
            # saving the model after every epoch
            save_trained_model(color, model, last_epoch + epoch + 1, optimizer, path)

    # saving the model after it finished training
    # save_trained_model(color, model, last_epoch + epochs, optimizer, path)

    # testing the model after all epochs
    if model.recurrent is True:
        # test_rnn(test_dataloader, model, device)
        print("Testing for rnn currently unavailable!")
        pass
    else:
        test_cnn(test_dataloader, model, device)

    print("Done!")


def generate_move(color, fen, amount_outputs=1):
    """
    When using the AI this function will return the move for a given
    game state.
    :param color: what type of AI is to be trained
    :param fen: current game state
    :param amount_outputs: the top amount of targets to be returned
    """
    # TODO: add mechanism for server to flexibly choose model/ and reuse same rnn model
    # loading the model
    state, path = load_model('recurrent_conv_black.pth')
    model = models.init_neural_network(state['output_size'], models.NOPOOL_BIGFC_LAYER)
    model.load_state_dict(state['model_state_dict'], strict=False)
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
        pred, _ = model(x)
        pred = dt.tensor_to_targets(pred,
                                    outputs,
                                    amount_outputs)

        print(f'Predicted: "{pred}"')

    return pred


def load_model(path=None):
    """
    Loading a saved model (.pth)
    :param path: name of the default file for loading
    """

    # Getting color from user
    if path is None:
        while True:
            print('Please pick one of the following model states!')
            # Printing the name of each model state
            files = os.listdir('models/')
            for i, file in enumerate(files):
                print(f'{i}) {file}')

            option = input(f'Pick model for loading (0-{len(files)-1}): ')
            try:
                option = int(option)
            except ValueError:
                print(f'Expected value parsable to type {int} but recieved value {type(option)}!')
                pass

            if 0 <= option < len(files):
                path = files[option]
                break
            else:
                print(f'Option {option} is out of bounds!')
    else:
        path = path

    state = torch.load('models/'+path)

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
    models_path: str = "models"
    if not os.path.exists(models_path):
        os.mkdir(models_path)

    torch.save({
                'color': color,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'output_size': datasets.get_output_length(color)},
               f'{models_path}/{path}')
    print(f"Saved PyTorch Current Model State to {path}")


def get_training_device():
    """
    Looks for available torch devices and returns the highest available
    """

    device = (
        # firstly checks if cuda is available
        "cuda"
        if torch.cuda.is_available()
        # if not the cpu will be used for training
        else "cpu"
    )
    # debug message
    print(f"Currently using {device} device!")

    return device


def set_debug_apis(state: bool):
    """
    Disabling or enabling various debug APIs to enhance regular training without debugging / testing.
    https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    :param state: set APIs to True or False
    """
    torch.autograd.set_detect_anomaly(state)
    torch.autograd.profiler.emit_nvtx = state
    # The following docs elaborate on the usage of the profiler
    # https://pytorch.org/docs/stable/profiler.html
    torch.autograd.profiler.profile = state
