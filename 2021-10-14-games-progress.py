#!/usr/bin/env python3

import pickle

game_board_pairs = []
with open(
    "2021-10-12-mcts-v3-boards14-games.pickle", "rb"
) as f:
    try:
        while True:
            game_board_pairs.append(pickle.load(f))
    except EOFError:
        pass

print(len(game_board_pairs))
