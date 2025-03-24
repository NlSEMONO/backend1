from Game import Game, _to_matrix, _to_num, get_move
from numba import njit
import numba
import numpy as np
import sys
from datetime import datetime

ones = np.ones((8, 8))

if __name__ == '__main__':
    g = Game()
    lb = int(sys.argv[1])
    ub = int(sys.argv[2])
    to_example = np.array(
    # [
    #     [True, True, False, True, False, True, True, True],
    #     [True, True, False, True, False, True, True, True],
    #     [True, True, False, False, False, False, False, False], 
    #     [False, False, False, False, False, False, False, False],
    #     [True, False, True, True, True, True, True, True],
    #     [False, False, False, False, False, False, False, False],
    #     [True, True, True, True, True, False, False, False],
    #     [True, True, True, True, True, False, False, False]
    # ]
    [
        [False, False, True, True, True, True, True, True],
        [True, False, True, True, True, True, True, True],
        [True, False, True, True, True, True, True, True],
        [True, False, False, False, False, False, False, False],
        [True, False, False, False, False, False, False, False],
        [True, False, False, False, False, False, False, False],
        [True, False, False, False, False, False, False, False],
        [True, False, False, True, True, True, True, True]
    ]
    )
    print(_to_num(to_example))
    for i in range(lb, ub):
        matrix = to_example
        get_move(matrix, [3, 25, 26], 2)
