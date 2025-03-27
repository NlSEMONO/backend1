from Game import Game, _to_matrix, _to_num, get_move, _print_matrix
from numba import njit
import numpy as np
import sys
from datetime import datetime
import os

ones = np.ones((8, 8))

if __name__ == '__main__':
    g = Game()
    lb = int(sys.argv[1])
    ub = int(sys.argv[2])
    to_example = np.array(
    # [
    #     [False, False, False, False, False, False, False, False],
    #     [False, False, False, False, False, False, False, False],
    #     [False, False, False, False, False, False, False, False],
    #     [False, False, False, False, False, False, False, False],
    #     [False, False, False, False, False, False, False, False],
    #     [False, False, False, False, False, False, False, False],
    #     [False, False, False, False, False, False, False, False],
    #     [False, False, False, False, False, False, False, False],
    # ]
    [
        [False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False],
    ]
    )
    combo = 2
    matrix = _to_matrix(17077909275448727849)
    _print_matrix(matrix, (0, 0, 0))
    while True:
        print(f"{_to_num(matrix)}")
        print("Enter blocks: ")
        choices = [int(x) for x in input().split()]
        combo = get_move(matrix, choices, combo)
        