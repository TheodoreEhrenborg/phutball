#!/usr/bin/env python3

import main
import random
import time
import pickle

player = main.MCTSPlayer3(
    num_evals=100000,
    curiosity=0.001,
    quiet=True,
    name="saved_model/2021-08-23-v3-model",
    pickle_in="2021-09-18-mcts-v3-boards13-random.pickle",
)


b = main.Board()

count = 0

while True:
    count += 1
    b.randomize(random.randint(0, 30))
    #    b.pretty_print_details()
    with open(
        "2021-09-22-malfunctioning-board.pickle", "wb"
    ) as f:
        pickle.dump(b, f)
    player.make_move(b.copy())
    if count % 10 == 0:
        print(count, "boards at ", time.strftime("%c"))
