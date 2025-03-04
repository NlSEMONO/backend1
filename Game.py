class Game:
    R_BOUND = 8
    L_BOUND = 0
    matrix = [[False for _ in range(8)] for _ in range(8)]
    _is_done = False
    true_symbol='#'
    false_symbol='.'
    blocks = [
        [[True for _ in range(5)]] + [[False for _ in range(5)]] * 4, # 1x5
        [[True] + [False] * 4] * 5, # 5 x 1
        [[True, True, True, False, False] * 3] + [[False for _ in range(5)]] * 2
    ]
    
    def __init__(self):
        pass
    
    def print_matrix(self):
        for row in self.matrix:
            print(' '.join(self.true_symbol if cell else self.false_symbol for cell in row))
    
    def print_blocks(self):
        print(self.blocks)
    
    def is_done(self):
        return self._is_done
    
    def _validate_action(self, b, x, y):
        if x < self.L_BOUND or x >= self.R_BOUND or y < self.L_BOUND or y >= self.R_BOUND:
            print("Invalid coordinates.")
            return False
        print(self.blocks[b])
        for i in range(5):
            for j in range(5):
                if not self.blocks[b][i][j]:
                    continue
                elif i + x >= self.R_BOUND or j + y >= self.R_BOUND:
                    print("Could not place block there :(")
                    return False
                
        return True
    
    def _place_block(self, b, x, y):
        for i in range(5):
            for j in range(5):
                if self.blocks[b][i][j]:
                    self.matrix[i+x][j+y] = True
                    # self.print_matrix()
    
    def _remove_rows_and_cols(self):
        # track rows and cols removed
        r = []
        c = []
        for i in range(self.R_BOUND):
            if all(self.matrix[i]):
                r.append(i)
        for i in range(self.R_BOUND):
            if all([self.matrix[j][i] for j in range(self.R_BOUND)]):
                c.append(i)

        # reset rows and cols
        # print(f'{r} {c}')
        for row in r:
            for i in range(self.R_BOUND):
                self.matrix[row][i] = False
                
        for col in c:
            for i in range(self.R_BOUND):
                self.matrix[i][col] = False
                
    
    def perform_action(self, b, x, y):
        if self._validate_action(b, x, y):
            self._place_block(b, x, y)
            self._remove_rows_and_cols()
        