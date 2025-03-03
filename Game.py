class Game:
    matrix = [[False] * 8]
    def print_matrix(matrix, true_symbol='#', false_symbol='.'):
        for row in matrix:
            print(' '.join(true_symbol if cell else false_symbol for cell in row))
