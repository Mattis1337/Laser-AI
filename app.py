# importing libraries
# import numpy as np
import chess as c

# importing own files
import datasets
import train_model

print(r' __          ___           _______. _______ .______               ___       __ ' + '\n' +
      r'|  |        /   \         /       ||   ____||   _  \             /   \     |  |' + '\n' +
      r'|  |       /  ^  \       |   (----`|  |__   |  |_)  |    ______ /  ^  \    |  |' + '\n' +
      r'|  |      /  /_\  \       \   \    |   __|  |      /    |______/  /_\  \   |  |' + '\n' +
      r'|  `----./  _____  \  .----)   |   |  |____ |  |\  \----.     /  _____  \  |  |' + '\n' +
      r'|_______/__/     \__\ |_______/    |_______|| _| `._____|    /__/     \__\ |__|' + '\n')

print("Laser-AI")

# TODO: https://discuss.pytorch.org/t/how-to-load-a-dpretrained-model-with-a-different-output-dimension/26117/7
# configure the loading of changed output sizes
# TODO: uncomment everything run twice and the its done

dataset = datasets.init_chess_dataset(c.BLACK)

board = c.Board()
board.push_san("e4")

train_model.train_chess_model(dataset, 100)
train_model.generate_move(c.BLACK, board.fen())

print("Goodbye")
