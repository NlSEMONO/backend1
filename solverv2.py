from Game import Game, _unique_paths, _to_matrix
from numba import njit
import numba
import numpy as np
import sys
from datetime import datetime

if __name__ == '__main__':
    g = Game()
    lb = int(sys.argv[1])
    ub = int(sys.argv[2])
    for i in range(lb, ub):
        matrix = _to_matrix(i)
        a = datetime.now()
        _unique_paths(matrix)
        print(datetime.now() - a)
