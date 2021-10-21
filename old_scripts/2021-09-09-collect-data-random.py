#!/usr/bin/env python3

import main
import random
import sys
import time

sys.path.insert(0, "/Users/theodore/other_python_programs")

import is_daytime

player = main.MCTSPlayer3(
    num_evals=10000,
    curiosity=0.001,
    quiet=True,
    name="saved_model/2021-08-23-v3-model",
    pickle_in="2021-09-09-mcts-v3-boards10-random.pickle",
)

b = main.Board()

count = 0

while True:
    while not is_daytime.is_daytime():
        count += 1
        b.randomize(random.randint(0, 30))
        player.make_move(b.copy())
        if count % 100 == 0:
            print(count, "boards")
    time.sleep(60)
