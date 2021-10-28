#!/usr/bin/env python3

import main
import random
import time

player = main.MCTSPlayer3(
    num_evals=100000,
    curiosity=0.001,
    quiet=True,
    name="saved_model/2021-10-26-v7-model",
    pickle_in="2021-10-26-mcts-v7-boards15-random.pickle",
)

b = main.Board()

count = 0

while True:
    count += 1
    b.randomize(random.randint(0, 30))
#    b.pretty_print_details()
    player.make_move(b.copy())
    if False and count % 10 == 0:
        print(count, "boards at ", time.strftime("%c"))
