import random

type grid = list[list[bool]]
type turn = tuple[int, int, int]

class Game:
    R_BOUND = 8
    L_BOUND = 0
    matrix = [[False for _ in range(8)] for _ in range(8)]
    _is_done = False
    true_symbol='#'
    false_symbol='.'
    choices = [0, 0, 0]
    perms_2 = [
        [[True for _ in range(5)]], # 1x5
        [[True for _ in range(4)]], # 1x4
        [[True for _ in range(3)]], # 1x3
        [[True for _ in range(2)]], # 1x2
        [[j == i for j in range(3)] for i in range(3)], # staircase
        [[True, False], [True, True], [False, True]], # Z-shape 0
        [[False, True], [True, True], [True, False]], # Z-shape reflect 0
        [[True for _ in range(3)] for _ in range(2)], # 2x3
        
    ]
    perms_4 = [
        [[True, True, True], [False, True, False]], # T-shape 0
        [[True, False], [True, False], [True, True]], # L-shape 0
        [[False,True], [False,True], [True, True]], # L-shape reflect 0
        [[True, False, False], [True, False, False], [True, True, True]], # L-shape big 0
        
    ]
    blocks = [
        [[]], # padding
        [[True, True, True]] * 3, # 3x3
        [[True, True]] * 2, # 2x2
    ]
    all_blocks = 0
    
    def _add_rotations(self, block: grid, r: int):
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
    
    def _print_matrix(self):
        c = 0
        for row in self.matrix:
            print(f"{c + 1} " + ' '.join(self.true_symbol if cell else self.false_symbol for cell in row))
            c += 1
        print(" ", end="")
        for i in range(self.R_BOUND):
            print(f" {i + 1}", end="")
        print()
    
    def _print_line(self):
        print()
    
    def _print_blocks(self):
        m = max([len(self.blocks[c]) for c in self.choices])
        size = 6
        for i in range(m):
            print_str = ""
            for c in self.choices:
                if len(self.blocks[c]) > i: 
                    width = len(self.blocks[c][0])
                    buf = size - width
                    print_str += " " * (buf // 2)
                    for j in range(width):
                        print_str += self.true_symbol if self.blocks[c][i][j] else self.false_symbol
                    print_str += " " * ((buf // 2) + (buf % 2)) + "|"
                else:
                    print_str += " " * (size) + "|"
            print(print_str)
        print_str = ""
        for i in range(len(self.choices)):
            buf = size
            print_str += " " * (buf // 2) + f"{i + 1}" + " " * ((buf // 2) + (buf % 2))
        print(print_str)
    
    def is_done(self):
        return self._is_done
    
    def _remove_block(self, b: int, x: int, y: int, matrix: grid):
        n = len(self.blocks[b])
        m = len(self.blocks[b][0])
        for i in range(n):
            for j in range(m): 
                if self.blocks[b][i][j]:
                    matrix[i+x][j+y] = False
    
    def _place_block(self, b: int, x: int, y: int, matrix: grid):
        n = len(self.blocks[b])
        m = len(self.blocks[b][0])
        for i in range(n):
            for j in range(m):
                if self.blocks[b][i][j]:
                    matrix[i+x][j+y] = True
    
    def _validate_action(self, b: int, x: int, y: int, matrix: grid, print_error = True):
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
    
    def _remove_rows_and_cols(self, matrix: grid):
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
    
    def _add_rows_and_cols(self, r: int, c: int , m: grid):
        for row in r:
            for i in range(self.R_BOUND):
                m[row][i] = True
        for col in c:
            for i in range(self.R_BOUND):
                m[i][col] = True
    
    def _gen_perms(self, nums: list[int], perm_so_far: list[int], used: set[int], res: set[turn]):
        # valid permutation
        if len(nums) == len(perm_so_far):
            res.add(tuple(perm_so_far))
            return
        # recurse
        for i in range(len(nums)):
            if i not in used:
                perm_so_far.append(i)
                used.add(i)
                self._gen_perms(nums, perm_so_far, used, res)
                used.remove(i)
                perm_so_far.pop()
    
    def _validate_choice_wkr(self, blks: list[int], m: grid, idx: int):
        if idx == 3:
            return True
        for i in range(self.R_BOUND):
            for j in range(self.R_BOUND):
                if self._validate_action(blks[idx], i, j, m, False):
                    self._place_block(blks[idx], i, j, m)
                    r, c = self._remove_rows_and_cols(m)
                    self._validate_choice_wkr(blks, m, idx + 1)
                    self._add_rows_and_cols(r, c, m)
                    self._remove_block(blks[idx], i, j, m)
                    return True
        return False
    
    def _validate_choice(self, blks: list[int], m: grid, dp):
        perms = set()
        used = set()
        self._gen_perms(blks, [], used, perms)
        for perm in perms:
            if self._validate_choice_wkr(list(perm), m, 0):
                return True
        return False
        
    def _get_next_3(self):
        res = set()
        dp = [[[False for _ in range(self.all_blocks)] for _ in range(self.all_blocks)] for _ in range(self.all_blocks)]
        recur_matrix = [[self.matrix[i][j] for j in range(self.R_BOUND)] for i in range(self.R_BOUND)]
        
        # see if there is a placement of the permutation that does not result in death
        for i in range(1, self.all_blocks):
            for j in range(i, self.all_blocks):
                for k in range(j, self.all_blocks):
                    if self._validate_choice([i, j, k], recur_matrix, dp):
                        res.add(tuple([i, j, k]))
        res = list(res)
        choice = random.choice(res)
        self.choices = list(choice)
        
    def perform_action(self, b: int, x: int, y: int):
        b -= 1
        x -= 1
        y -= 1
        if b >= 3 or b < 0:
            print("Invalid block selected")
            return
        
        if self._validate_action(self.choices[b], x, y, self.matrix):
            self._place_block(self.choices[b], x, y, self.matrix)
            self.choices[b] = 0
            self._remove_rows_and_cols(self.matrix)
            
    
    def print_state(self):
        if all([c == 0 for c in self.choices]):
            self._get_next_3()
        self._print_matrix()
        self._print_line()
        self._print_blocks()
