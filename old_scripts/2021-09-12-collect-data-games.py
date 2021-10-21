#!/usr/bin/env python3


import sys
import time
import main

sys.path.insert(0, "/Users/theodore/other_python_programs")

import is_daytime

PICKLE_IN = "2021-09-12-mcts-v3-boards11-games.pickle"

MODEL_NAME = "saved_model/2021-08-23-v3-model"

left_player = main.MCTSPlayer3(
    num_evals=10000,
    curiosity=0.001,
    randomize=True,
    pickle_in=PICKLE_IN,
    quiet=True,
    name=MODEL_NAME,
)
right_player = main.MCTSPlayer3(
    num_evals=10000,
    curiosity=0.001,
    randomize=True,
    pickle_in=PICKLE_IN,
    quiet=True,
    name=MODEL_NAME,
)

count = 0

while True:
    while not is_daytime.is_daytime():
        count += 1
        main.run_game(
            quiet=True,
            truncate_time=1000,
            players=[left_player, right_player],
        )
        if count % 10 == 0:
            print(count, "games")
    time.sleep(60)
