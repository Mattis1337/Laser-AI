# importing libraries
# import numpy as np
import chess as c

# importing own files
import datasets
import train_model


print("Laser-AI")

# Training the model works as follows:
# usually you take a batch size of 100
for i in range(100):
    dataset = datasets.init_chess_dataset(c.BLACK, 16, i)  # batch size has to be 16

    train_model.train_chess_model(dataset)

print("Goodbye")
