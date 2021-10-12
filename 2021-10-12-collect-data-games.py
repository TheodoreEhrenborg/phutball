#!/usr/bin/env python3

import main

PICKLE_IN = "2021-10-12-mcts-v3-boards14-games.pickle"

MODEL_NAME = "saved_model/2021-08-23-v3-model"

left_player = main.MCTSPlayer3(
    num_evals=10 ** 5,
    curiosity=0.001,
    randomize=True,
    pickle_in=PICKLE_IN,
    quiet=True,
    name=MODEL_NAME,
)
right_player = left_player

count = 0

while True:
    count += 1
    main.run_game(
        quiet=True,
        truncate_time=2000,
        players=[left_player, right_player],
    )
    if count % 10 == 0:
        print(count, "games")
