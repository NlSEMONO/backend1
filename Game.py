from random import random

class Game:
    R_BOUND = 8
    L_BOUND = 0
    matrix = [[False for _ in range(8)] for _ in range(8)]
    _is_done = False
    true_symbol='#'
    false_symbol='.'
    perms_2 = [
        [[True for _ in range(5)]], # 1x5
        [[True for _ in range(4)]], # 1x4
        [[True for _ in range(3)]], # 1x3
        [[True, False], [True, True], [False, True]], # Z-shape 0
        [[False, True], [True, True], [True, False]], # Z-shape reflect 0
        [[True for _ in range(3)] for _ in range(2)], # 2x3
        
    ]
    perms_4 = [
        [[True, True, True], [False, True, False]], # T-shape 0
        [[True, False], [True, False], [True, True]], # L-shape 0
        [[False,True], [False,True], [True, True]], # L-shape reflect 0
        [[True, False, False], [True, False, False], [True, True, True]], # L-shape big 0
        [[False, False,True], [False, False,True], [True, True, True]], # L-shape big reflect 0
        
    ]
    blocks = [
        [[True, True, True]] * 3, # 3x3
        [[True, True]] * 2, # 2x2
    ]
    all_blocks = 0
    
    def _add_rotations(self, block: list[list[bool]], r):
        tmp = block
        self.blocks.append(block)
        for _ in range(r - 1):
            n = len(tmp)
            m = len(tmp[0])
            tmp2 = [[False for _ in range(n)] for _ in range(m)]
            for i in range(n):
                for j in range(m):
                    tmp2[j][n-i-1] = tmp[i][j]
            tmp = tmp2
            self.blocks.append(tmp)
    
    def __init__(self):
        for block in self.perms_2:
            self._add_rotations(block, 2)

        for block in self.perms_4:
            self._add_rotations(block, 4)
            
        self.all_blocks = len(self.blocks)
    
    def print_matrix(self):
        for row in self.matrix:
            print(' '.join(self.true_symbol if cell else self.false_symbol for cell in row))
    
    def print_blocks(self):
        print(self.blocks)
    
    def is_done(self):
        return self._is_done
    
    def _validate_action(self, b, x, y, matrix, print_error = True):
        if x < self.L_BOUND or x >= self.R_BOUND or y < self.L_BOUND or y >= self.R_BOUND:
            if (print_error):
                print("Invalid coordinates.")
                return False
        n = len(self.blocks[b])
        m = len(self.blocks[b][0])
        for i in range(n):
            for j in range(m):
                if not self.blocks[b][i][j]:
                    continue
                elif i + x >= self.R_BOUND or j + y >= self.R_BOUND:
                    if (print_error):
                        print("Could not place block there :(")
                    return False
                elif matrix[i+x][j+y] and self.blocks[b][i][j]:
                    if print_error:
                        print("Space already occupied :(")
                    return False
                
        return True
    
    def _remove_block(self, b, x, y, matrix):
        n = len(self.blocks[b])
        m = len(self.blocks[b][0])
        for i in range(n):
            for j in range(m): 
                if self.blocks[b][i][j]:
                    matrix[i+x][j+y] = False
    
    def _place_block(self, b, x, y, matrix):
        n = len(self.blocks[b])
        m = len(self.blocks[b][0])
        for i in range(n):
            for j in range(m):
                if self.blocks[b][i][j]:
                    matrix[i+x][j+y] = True
                    # self.print_matrix()
    
    def _remove_rows_and_cols(self, matrix):
        # track rows and cols removed
        r = []
        c = []
        for i in range(self.R_BOUND):
            if all(matrix[i]):
                r.append(i)
        for i in range(self.R_BOUND):
            if all([matrix[j][i] for j in range(self.R_BOUND)]):
                c.append(i)

        # reset rows and cols
        for row in r:
            for i in range(self.R_BOUND):
                matrix[row][i] = False
                
        for col in c:
            for i in range(self.R_BOUND):
                matrix[i][col] = False
                
        return [r, c]
    
    def _add_rows_and_cols(self, r, c, m):
        for row in r:
            for i in range(self.R_BOUND):
                m[row][i] = True
        for col in c:
            for i in range(self.R_BOUND):
                m[i][col] = True
                
    def _next_3_wkr_gen(self, res, arr, m, idx):
        if len(arr) == 3:
            res.add(tuple(arr))
            return True
        else:
            live = False
            for k in range(idx, self.all_blocks):
                for i in range(self.R_BOUND):
                    for j in range(self.R_BOUND):
                        # if placing block on valid index, see if it is possible to form a valid group of 3 w/o death
                        if self._validate_action(k, i, j, m, False):
                            self._place_block(k, i, j, m)
                            r, c = self._remove_rows_and_cols(m)
                            arr.append(k)
                            rr = self._next_3_wkr_gen(res, arr, m, k)
                            arr.pop()
                            self._add_rows_and_cols(r, c, m)
                            self._remove_block(k, i, j, m)
                            live = rr or live
                            
                            # jump to next block if current block has valid group of 3
                            if rr:
                                with open("try_log.log", 'a') as f:
                                    f.write(f'{i * 8 + j} tries')
                                i = self.R_BOUND
                                j = self.R_BOUND
            return live
                    
    def _get_next_3(self):
        res = set()
        recur_matrix = [[self.matrix[i][j] for j in range(self.R_BOUND)] for i in range(self.R_BOUND)]
        self._next_3_wkr_gen(res, [], recur_matrix, 0)
        random.choice(res)
        
    def perform_action(self, b, x, y):
        if self._validate_action(b, x, y, self.matrix):
            self._place_block(b, x, y, self.matrix)
            self._remove_rows_and_cols()
        