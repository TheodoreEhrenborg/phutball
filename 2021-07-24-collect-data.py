#!/usr/bin/env python3

import main
import random
import sys
import time

sys.path.insert(0, "/Users/theodore/other_python_programs")

import is_daytime

player = main.NegamaxABPlayer(
    depth=2,
    pickle_in="2021-07-24-3ply-boards4.pickle",
    quiet=True,
    static_evaluator=main.ParallelCNNEvaluator(
        name="saved_model/2021-07-22-model"
    ),
)


b = main.Board()

count = 0

while True:
    while not is_daytime.is_daytime():
        count += 1
        b.randomize(random.randint(0, 30))
        player.make_move(b.copy())
        if count % 10 == 0:
            print(count, "boards")
    time.sleep(60)
