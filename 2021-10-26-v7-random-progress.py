#!/usr/bin/env python3

import pickle

board_pairs = []
with open(
    "2021-10-26-mcts-v7-boards15-random.pickle", "rb"
) as f:
    try:
        while True:
            board_pairs.append(pickle.load(f))
    except EOFError:
        pass

print(len(board_pairs))
