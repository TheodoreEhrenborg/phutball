#!/usr/bin/env python3


import sys
import time
import main

sys.path.insert(0, "/Users/theodore/other_python_programs")

import is_daytime

left_player = main.NegamaxABPlayer(
    depth=2,
    pickle_in="2021-08-04-3ply-boards5.pickle",
    top_few=2,
    quiet=True,
    static_evaluator=main.ParallelCNNEvaluator(
        name="saved_model/2021-07-22-model"
    ),
)

right_player = main.NegamaxABPlayer(
    depth=2,
    pickle_in="2021-08-04-3ply-boards5.pickle",
    top_few=2,
    quiet=True,
    static_evaluator=main.ParallelCNNEvaluator(
        name="saved_model/2021-07-22-model"
    ),
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
