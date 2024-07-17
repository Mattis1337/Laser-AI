# importing libraries
# import numpy as np
import chess as c

# importing own files
import datasets
import train_model


print("Laser-AI")

dataset = datasets.init_chess_dataset(c.BLACK)

train_model.train_chess_model(dataset)

print("Goodbye")
