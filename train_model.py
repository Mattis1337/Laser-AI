import os.path

import chess
import chess as c
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from random import shuffle

# importing own files
import chess_annotation
import datasets
import data_transformations as dt
import models

OUTPUTS_WHITE: list[str] = dt.targets_to_numericals(c.WHITE)
OUTPUTS_BLACK: list[str] = dt.targets_to_numericals(c.BLACK)


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

        try:
            col_index = int(input('Pick an option 1-2:'))
            if col_index in [1, 2]:
                break
            print(f'Specified option {col_index} is invalid!')
        except ValueError:
            print(f'Please only use integer values as input!')
            pass

    if col_index == 1:
        color = chess.BLACK
    else:
        color = chess.WHITE
    # getting the outputs matching the color for the topology
    outputs = datasets.get_output_length(color)
    # setting the fitting topology of an untrained network
    model = models.init_neural_network(outputs)

    if hasattr(model, 'recurrent'):
        if model.recurrent is True:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # entering new path to save the model to
    path = input('Insert name for the new model state (.pth will be appended!): ')

    while True:
        ans = int(input('Would you like to warmstart the model using already pr trained weights? (y=0, n=1)'))
        try:
            if ans not in [0, 1]:
                print('Invalid answer! (0/1 only)')
                continue
            if ans == 0:
                state, pre_path = load_model(None)
                model.load_state_dict(torch.load('models/'+pre_path, weights_only=True), strict=False)
                optimizer.load_state_dict(state['optimizer_state_dict'])
            break
        except ValueError:
            print(f'Please only use integer values as input!')
            pass

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
        pred = model(inputs.to(device))
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


def train_rnn(dataset, model, criterion, optimizer, device):
    """
    Training a recurrent neural network
    :param dataset: the dataset containing the white/black moves which shall be used for training
    :param model: a model object instantiated from NeuralNetwork
    :param criterion: loss function
    :param optimizer: the optimizer used for enhancing the training algorithm
    :param device: the device currently used for training
    """
    running_loss = 0
    total_states = 0
    model.zero_grad()

    order = [[i] for i in range(len(dataset))]
    print(f'Training on {len(dataset.transformed_games)} randomly shuffled games!')
    shuffle(order)

    for batch, idx in enumerate(order):
        input_sequence, target_sequence = dataset.__getitem__(idx)
        input_sequence.unsqueeze(-1)
        if input_sequence.size(0) == 0:
            continue
        input_sequence.to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        loss = torch.Tensor([0])  # you can also just simply use ``loss = 0``

        input_sequence.squeeze(1)
        # iterating through all previous moves
        for i in range(input_sequence.size(0)):
            total_states += 1
            output = model(input_sequence[i].to(device))
            l = criterion(output.to(device), target_sequence[i].to(device))
            loss += l

        # if the loss is scalar it can not be handled by the back propagation
        if loss.requires_grad is not False:
            loss.backward()
            optimizer.step()
        else:
            print("passing")
        running_loss += loss

        if batch % 1000 == 0:
            print(f"average loss: {running_loss/total_states}  [{batch:>5d}/{len(dataset):>5d}]")
            running_loss = 0
            total_states = 0

# other approach
# loss = torch.Tensor([0])  # you can also just simply use ``loss = 0``
#        past_inputs = []
#        for seq_idx in range(input_sequence.size(0)):
#            past_inputs.append(input_sequence[seq_idx])
#            if len(past_inputs) > MAX_SEQ_LEN:
#                past_inputs.pop(0)
#           inputs = torch.stack(past_inputs)
#            print(inputs.size())
#            total_states += 1
#            output = model(inputs.to(device))
#            l = criterion(output.to(device), torch.unsqueeze(target_sequence[seq_idx], 0).to(device))
#            loss += l
#
#            # if the loss is scalar it can not be handled by the back propagation
#            if loss.requires_grad is not False:
#                loss.backward()
#               optimizer.step()
#            else:
#               print("passing")
#            running_loss += loss


MAX_SEQ_LEN = 16


def train_lstm(dataset, criterion, model, optimizer, device):
    """Training an lstm"""
    running_loss = 0
    total_states = 0
    model.zero_grad()

    order = [[i] for i in range(len(dataset))]
    print(f'Training on {len(dataset.transformed_games)} randomly shuffled games!')
    shuffle(order)

    for batch, idx in enumerate(order):
        input_sequence, target_sequence = dataset.__getitem__(idx)
        # input_sequence = torch.zeros([42, 12, 8, 8])
        # target_sequence = torch.zeros([42, 1863])

        if input_sequence.size(0) == 0:
            continue

        # to address the vanishing gradient problem we separate the data into chunks of a max size of MAX_SEQ_LEN
        chunks = int(np.ceil(input_sequence.size(0) / MAX_SEQ_LEN))
        chunked_input = []
        chunked_targets = []
        for c in range(chunks):
            if c == chunks-1:
                chunked_input.append(input_sequence[c*MAX_SEQ_LEN:, :, :, :])
                chunked_targets.append(target_sequence[c*MAX_SEQ_LEN:, :])
            chunked_input.append(input_sequence[c*MAX_SEQ_LEN:(c+1)*MAX_SEQ_LEN, :, :, :])
            chunked_targets.append(target_sequence[c*MAX_SEQ_LEN:(c+1)*MAX_SEQ_LEN, :])

        for chunk_idx in range(chunks):
            input_sequence = chunked_input[chunk_idx]
            target_sequence = chunked_targets[chunk_idx]
            target_sequence = torch.unsqueeze(target_sequence, 0)
            optimizer.zero_grad(set_to_none=True)

            loss = torch.Tensor([0])  # you can also just simply use ``loss = 0``

            total_states += 1
            output = model(input_sequence.to(device))

            l = criterion(output.to(device), target_sequence.to(device))
            loss += l

            # if the loss is scalar it can not be handled by the back propagation
            if loss.requires_grad is not False:
                loss.backward()
                optimizer.step()
            else:
                print("passing")

            running_loss += loss

        if batch % 1000 == 0:
            print(f"average loss: {running_loss/total_states}  [{batch:>5d}/{len(dataset):>5d}]")
            running_loss = 0
            total_states = 0


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
            pred = model(image.to(device))
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


def test_rnn(dataset, model, device):
    """
    Testing the accuracy of a given recurrent model by calculating the total error in a given output by averaging the error per
    digit in an output and adding it.
    :param dataset: the dataset containing the white/black moves which shall be used for training
    :param model: a model object instantiated from NeuralNetwork
    :param device: the device currently used for testing
    """

    model.eval()
    # enabling oneDNN graphs for better performance
    torch.jit.enable_onednn_fusion(True)

    total = 0
    n_total = 0
    order = [[j] for j in range(len(dataset))]
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for batch, idx in enumerate(order):
            input_sequence, target_sequence = dataset.__getitem__(idx)

            if input_sequence.size(0) == 0:
                print(f"Skipping malformed data: {input_sequence.size()} (dimension)")
                continue
            # getting rid of the batch dimension which effectively is 1 all the time
            input_sequence.unsqueeze(-1)

            # calculate outputs by running game states through the network
            for seq_idx in range(len(input_sequence)):
                output = model(input_sequence[seq_idx].to(device))

                # getting the highest match of the AI output
                pred = dt.get_highest_index(output, 1)
                target = dt.get_highest_index(target_sequence[seq_idx], 1)
                n_total += 1

                # comparing if the AI's match is the same as the output
                if int(target[0]) == int(pred[0]):
                    total += 1

            if (batch+1) % 1000 == 0:
                print(f'Successfully tested {n_total} randomly shuffled game states ({batch+1} games)!')
                break

    print(f'Accuracy of the network: {(total/n_total)*100}%')
    print(f'Predicted {total} moves out of {n_total} total moves correctly!')
    model.train()


def test_lstm(dataset, model, device):
    """
    Testing the accuracy of a given recurrent model by calculating the total error in a given output by averaging the error per
    digit in an output and adding it.
    :param dataset: the dataset containing the white/black moves which shall be used for training
    :param model: a model object instantiated from NeuralNetwork
    :param device: the device currently used for testing
    """

    model.eval()
    # enabling oneDNN graphs for better performance
    torch.jit.enable_onednn_fusion(True)

    total = 0
    n_total = 0
    order = [[j] for j in range(len(dataset))]
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for batch, idx in enumerate(order):
            input_sequence, target_sequence = dataset.__getitem__(idx)

            if input_sequence.size(0) == 0:
                print(f"Skipping malformed data: {input_sequence.size()} (dimension)")
                continue

            # to address the vanishing gradient problem we separate the data into chunks of a max size of MAX_SEQ_LEN
            chunks = int(np.ceil(input_sequence.size(0) / MAX_SEQ_LEN))
            chunked_input = []
            chunked_targets = []
            for c in range(chunks):
                if c == chunks - 1:
                    chunked_input.append(input_sequence[c * MAX_SEQ_LEN:, :, :, :])
                    chunked_targets.append(target_sequence[c * MAX_SEQ_LEN:, :])
                chunked_input.append(input_sequence[c * MAX_SEQ_LEN:(c + 1) * MAX_SEQ_LEN, :, :, :])
                chunked_targets.append(target_sequence[c * MAX_SEQ_LEN:(c + 1) * MAX_SEQ_LEN, :])

            for chunk_idx in range(chunks):
                input_sequence = chunked_input[chunk_idx]
                target_sequence = chunked_targets[chunk_idx]

                output = model(input_sequence.to(device))
                output = torch.squeeze(output, 0)

                for pred_idx in range(len(output)):
                    # getting the highest match of the AI output
                    pred = dt.get_highest_index(output[pred_idx], 1)
                    target = dt.get_highest_index(target_sequence[pred_idx], 1)
                    n_total += 1

                    # comparing if the AI's match is the same as the output
                    if int(target[0]) == int(pred[0]):
                        total += 1

            if (batch+1) % 1000 == 0:
                print(f'Successfully tested {n_total} randomly shuffled game states ({batch+1} games)!')
                break

    print(f'Accuracy of the network: {(total/n_total)*100}%')
    print(f'Predicted {total} moves out of {n_total} total moves correctly!')
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

    # setting the number of usable threads
    # USE AT OWN RISK
    torch.set_num_threads(8)

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
    if hasattr(model, 'recurrent'):
        if model.recurrent is True:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate*0.01, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate*0.1)
    # checking if the output size has changed in between learning
    if old_outputs != datasets.get_output_length(color):
        # change the dimension of the output
        if isinstance(model, models.NeuralNetwork):
            num_ftrs = model.out_fc.in_features
            model.out_fc = nn.Linear(num_ftrs, old_outputs)
            # loading the pre-trained model with the old outputs
            model.load_state_dict(state['model_state_dict'], strict=False)
            # changing the model to new output size
            model.out_fc = nn.Linear(num_ftrs, datasets.get_output_length(color))
        else:
            num_ftrs = model.fc3.in_features
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

    if hasattr(model, 'recurrent'):
        if model.recurrent is True:
            # initializing RNN dataset
            dataset = datasets.init_chess_dataset(color, True)
            # changing memory format for RNN models
            model.to(memory_format=torch.channels_last)
        else:
            # initializing the dataset/dataloader for CNN models
            dataset = datasets.init_chess_dataset(color, False)
            train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)
    else:
        criterion = nn.CrossEntropyLoss()
        # initializing RNN dataset
        dataset = datasets.init_chess_dataset(color, True)

    # Printing info
    print(f'Resuming training at epoch {last_epoch}!')
    # Number of trainable parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # amount of iterations after which a new random sampled dataset should be created
    same_sample_iters = 5

    # Train the network for the set epoch size
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} (total {last_epoch+epoch+1})\n-------------------------------")
        if hasattr(model, 'recurrent'):
            if model.recurrent is True:
                if epoch % same_sample_iters == 0 and epoch != 0:
                    # switching up the loaded samples
                    dataset.__sample__()
                train_rnn(dataset, model, criterion, optimizer, device)
            else:
                train_cnn(train_dataloader, model, criterion, optimizer, device)
        else:
            if epoch % same_sample_iters == 0 and epoch != 0:
                # switching up the loaded samples
                dataset.__sample__()
            train_lstm(dataset, criterion, model, optimizer, device)

        if (epoch+1) % 1 == 0:
            # saving the model after every epoch
            save_trained_model(color, model, last_epoch + epoch + 1, optimizer, path)  # + epoch + 1
            pass

    # saving the model after it finished training
    # save_trained_model(color, model, last_epoch + epochs, optimizer, path)

    # testing the model after all epochs
    if hasattr(model, 'recurrent'):
        if model.recurrent is True:
            # test_dataloader = DataLoader(dataset, shuffle=False, num_workers=8)
            test_rnn(dataset, model, device)
            pass
        else:
            test_dataloader = DataLoader(dataset, shuffle=False, num_workers=8)
            test_cnn(test_dataloader, model, device)
    else:
        test_lstm(dataset, model, device)
        pass
    print("Done!")


SEQUENCED_INPUT = []


def generate_move(color, fen, amount_outputs=None):
    """
    When using the AI this function will return the move for a given
    game state.
    :param color: what type of AI is to be trained
    :param fen: current game state
    :param amount_outputs: the top amount of targets to be returned
    """
    # TODO: add mechanism for server to flexibly choose model/ and reuse same rnn model
    # loading the model
    if color is chess.WHITE:
        state, path = load_model('white_cnn.pth')
    elif color is chess.BLACK:
        state, path = load_model('nopool_nopad_black.pth')
    else:
        raise ValueError(type(color) + " is not of type " + type(bool))

    if amount_outputs is None:
        amount_outputs = datasets.get_output_length(color)
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
    if hasattr(model, 'recurrent'):
        if model.recurrent is True:
            if SEQUENCED_INPUT is None:
                SEQUENCED_INPUT.append(torch.squeeze(x, 0))
            else:
                x = torch.squeeze(x, 0)
                SEQUENCED_INPUT.append(x)
                x = torch.stack(SEQUENCED_INPUT)
    else:
        if SEQUENCED_INPUT is None:
            SEQUENCED_INPUT.append(torch.squeeze(x, 0))
        else:
            x = torch.squeeze(x, 0)
            SEQUENCED_INPUT.append(x)
            x = torch.stack(SEQUENCED_INPUT)

    outputs: list[str]
    match color:
        case c.WHITE:
            outputs = OUTPUTS_WHITE
        case c.BLACK:
            outputs = OUTPUTS_BLACK
        case _:
            raise ValueError("Expected variable color to be of type bool but received" + type(color))

    with torch.no_grad():
        pred = model(x)
        pred = [pred[-1]]
        pred = dt.tensor_to_targets(pred[0],
                                    outputs,
                                    amount_outputs)

        # print(f'Predicted: "{pred}"')

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

    state = torch.load('models/'+path, weights_only=True)

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
    torch.autograd.profiler.emit_nvtx(state)
    # The following docs elaborate on the usage of the profiler
    # https://pytorch.org/docs/stable/profiler.html
    torch.autograd.profiler.profile(state)

