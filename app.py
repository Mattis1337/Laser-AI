# importing libraries
import numpy as np

# importing own files
from annotation_converter import *
#import train_model

print("Laser-AI")

print("\n")
bitboard = fen_to_bitboard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

print_bitboard_fen(bitboard)
