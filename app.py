# importing libraries
# import numpy as np
import chess as c

# importing own files
import datasets
import train_model


print("Laser-AI")

# Training the model works as follows:
# usually you take a batch size of 100
dataset = datasets.init_chess_dataset(c.BLACK, 20, 1)

train_model.train_chess_model(dataset)
