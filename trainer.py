#!/usr/bin/env python

# importing libraries
# import numpy as np
# import chess
import sys

# importing own files
import train_model


def main():
    """
    Command Line User Interface
    """
    print("Script to train the Laser-AI!")

    while True:
        print(' 0) Exit')
        print(' 1) Train AI locally')
        print(' 2) Initialize new AI model')
        option = int(input('Pick an option 0-2: '))

        if option in range(0, 3):
            break
        print(f'Specified option {option} is invalid!')

    match option:
        case 0:
            sys.exit("Exiting...")
        case 1:
            # Starting the training process
            train_model.train_chess_model()
        case 2:
            # Starting the initialization process
            train_model.init_new_model()


if __name__ == "__main__":
    main()
    print("Goodbye")
