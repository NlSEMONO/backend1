"""
Contains methods to help simulate block blast and judge block blast moves
"""
import random

# Grid = game state
type Grid = list[list[bool]]
# Turn = choice of 3 blocks
type Turn = tuple[int, int, int]
# move = set of parameters to be fed into perform_action (think of this as placing a block)
type Move = tuple[int, int, int]
# Sequence = 3 moves defining the way each Turn block should be used to proceed to next Grid
type Sequence = tuple[Move, Move, Move]

class Game:
    """Methods to run a game of block blast
    """
    R_BOUND = 8
    L_BOUND = 0
    matrix = [[False for _ in range(8)] for _ in range(8)]
    _is_done = False
    true_symbol='#'
    false_symbol='.'
    choices = [0, 0, 0]
    _combo = 0
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
    
    def _adjust_combo(self, combo: int, r: list[int], c: list[int]):
        return 3 if len(r) + len(c) > 0 else max(0, combo - 1)
    
    def _add_rotations(self, block: Grid, r: int) -> None:
        """Adds the first r rotations of block to self.blocks

        Args:
            block (Grid): the block to be rotated
            r (int): number of 90 rotations to add to blocks list, starting at 0 degrees
        """
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
    
    def _print_matrix(self) -> None:
        c = 0
        for row in self.matrix:
            print(f"{c + 1} " + ' '.join(self.true_symbol if cell else self.false_symbol for cell in row))
            c += 1
        print(" ", end="")
        for i in range(self.R_BOUND):
            print(f" {i + 1}", end="")
        print()
    
    def _print_line(self) -> None:
        print()
    
    def _print_blocks(self) -> None:
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
        
    def _copy_matrix(self) -> Grid:
        """
        Returns a copy of self.matrix

        Returns:
            Grid: A copy of self.matrix
        """
        return [[self.matrix[i][j] for j in range(self.R_BOUND)] for i in range(self.R_BOUND)]
    
    def is_done(self) -> bool:
        """
        Returns:
            bool: Is the game done or not?
        """
        return self._is_done
    
    def _remove_block(self, b: int, x: int, y: int, matrix: Grid) -> None:
        """
        Args:
            b (int): block index inside self.blocks
            x (int): top left y-coordinate placement location of the block
            y (int): top left x-coordinate placement location of the block
            matrix (Grid): the matrix to remove the block from
        """
        n = len(self.blocks[b])
        m = len(self.blocks[b][0])
        for i in range(n):
            for j in range(m): 
                if self.blocks[b][i][j]:
                    matrix[i+x][j+y] = False
    
    def _place_block(self, b: int, x: int, y: int, matrix: Grid) -> None:
        """
        Args:
            b (int): block index to be placed
            x (int): top left y-coordinate placement location of the block
            y (int): top left x-coordinate placement location of the block
            matrix (Grid): the matrix to place the block into
        """
        n = len(self.blocks[b])
        m = len(self.blocks[b][0])
        for i in range(n):
            for j in range(m):
                if self.blocks[b][i][j]:
                    matrix[i+x][j+y] = True
    
    def _validate_action(self, b: int, x: int, y: int, matrix: Grid, print_error = True) -> bool:
        """
        Args:
            b (int): block index to be placed
            x (int): top left y-coordinate placement location of the block
            y (int): top left x-coordinate placement location of the block
            matrix (Grid): matrix to place the block into
            print_error (bool, optional): Print errors? Defaults to True

        Returns:
            bool: Valid to place block at (y, x) or not?
        """
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
    
    def _remove_rows_and_cols(self, matrix: Grid) -> list[list[int]]:
        """Removes completed rows and columns from matrix

        Args:
            matrix (Grid): matrix to remove rows and columns from

        Returns:
            list[list[int]]: the rows and columns that got removed
        """
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
    
    def _add_rows_and_cols(self, r: list[int], c: list[int] , m: Grid) -> None:
        """Opposite operation of _remove_rows_and_cols

        Args:
            r (list[int]): rows to add back
            c (list[int]): columns add back
            m (Grid): matrix to add rows/cols back into
        """
        for row in r:
            for i in range(self.R_BOUND):
                m[row][i] = True
        for col in c:
            for i in range(self.R_BOUND):
                m[i][col] = True
    
    def _gen_perms(self, nums: list[int], perm_so_far: list[int], used: set[int]) -> set[tuple[int, int, int]]:
        """
        Args:
            nums (list[int]): Turn as a list
            perm_so_far (list[int]): Accumulator variable for permutation so far
            used (set[int]): Set for tracking which numbers have already been used

        Returns:
            set[tuple[int, int, int]]: All possible orders to play the Turn
        """
        res = set()
        # valid permutation, add result
        if len(nums) == len(perm_so_far):
            res.add(tuple(perm_so_far))
            return res
        # recurse
        n = len(nums)
        for i in range(n):
            if i not in used:
                used.add(i)
                perm_so_far.append(nums[i])
                res = res.union(self._gen_perms(nums, perm_so_far, used))
                perm_so_far.pop()
                used.remove(i)
        return res
    
    def _validate_choice_wkr(self, blks: list[int], m: Grid, idx: int) -> bool:
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
    
    def _validate_choice(self, blks: list[int], m: Grid) -> bool:
        """
        Args:
            blks (list[int]): Turn to validate
            m (Grid): matrix to perform validation in

        Returns:
            bool: Does there exist a Sequence of moves to get through the Turn?
        """
        used = set()
        perms = self._gen_perms(blks, [], used)
        for perm in perms:
            if self._validate_choice_wkr(list(perm), m, 0):
                return True
        return False
    
    def _gen_liveable_choices(self, matrix: Grid) -> set[tuple[int, int, int]]:
        res = set()
        recur_matrix = [[matrix[i][j] for j in range(self.R_BOUND)] for i in range(self.R_BOUND)]
        
        # see if there is a placement of the permutation that does not result in death
        for i in range(1, self.all_blocks):
            for j in range(i, self.all_blocks):
                for k in range(j, self.all_blocks):
                    if self._validate_choice([i, j, k], recur_matrix):
                        res.add((i, j, k))
        return res
        
    def _get_next_3(self):
        res = list(self._gen_liveable_choices(self.matrix))
        choice = random.choice(res)
        self.choices = list(choice)
    
    def _get_sequences(self, perm: Turn, idx: int, m: Grid, seq_so_far: list[Move]) -> list[Sequence]:
        if idx == len(perm):
            return [tuple(seq_so_far)]
        options = []
        for i in range(self.R_BOUND):
            for j in range(self.R_BOUND):
                if self._validate_action(perm[idx], i, j, m, False):
                    self._place_block(perm[idx], i, j, m)
                    r, c = self._remove_rows_and_cols(m)
                    seq_so_far.append((perm[idx], i, j))
                    options.extend(self._get_sequences(perm, idx + 1, m, seq_so_far))
                    seq_so_far.pop()
                    self._add_rows_and_cols(r, c, m)
                    self._remove_block(perm[idx], i, j, m)
        return options
    
    def _get_placed_state(self, seq: Sequence, matrix: Grid) -> tuple[int, bool, list[tuple[list[int], list[int]]]]:
        tmp_combo = self._combo
        broke_combo = False
        rc_arr = []
        for move in seq:
            self._place_block(move[0], move[1], move[2], matrix)
            r, c = self._remove_rows_and_cols(matrix)
            tmp_combo = self._adjust_combo(tmp_combo, r, c)
            broke_combo = broke_combo or tmp_combo == 0
            rc_arr.append((r, c))
        return (tmp_combo, broke_combo, rc_arr)
    
    def _undo_placed_state(self, seq: Sequence, matrix: Grid, rc_arr: list[tuple[list[int], list[int]]]) -> None:
        for i in range(len(seq) - 1, -1, -1):
            self._add_rows_and_cols(rc_arr[i][0], rc_arr[i][1], matrix)
            self._remove_block(seq[i][0], seq[i][1], seq[i][2], matrix)
    
    def _combo_maintained(self, matrix: Grid, combo: int, perm: tuple[int, int, int], idx: int):
        if idx == 3:
            return True
        for i in range(self.R_BOUND):
            for j in range(self.R_BOUND):
                tmp_combo = combo
                if self._validate_action(perm[idx], i, j, matrix, False):
                    self._place_block(perm[idx], i, j, matrix)
                    r, c = self._remove_rows_and_cols(matrix)
                    tmp_combo = self._adjust_combo(tmp_combo, r, c)
                    if tmp_combo > 0:
                        if self._combo_maintained(matrix, tmp_combo, perm, idx + 1):
                            return True
                    self._add_rows_and_cols(r, c, matrix)
                    self._remove_block(perm[idx], i, j, matrix)
                    
        return False
    
    def _assess_state(self, matrix: Grid, combo: int):
        no_break = 0
        res = self._gen_liveable_choices(matrix)
        for choice in res:
            perms = self._gen_perms(list(choice), [], set())
            for perm in perms:
                if self._combo_maintained(matrix, combo, perm, 0):
                    no_break += 1
                    break
        return no_break / len(res)
                    
    
    def _get_best_turn(self):
        perms = self._gen_perms(self.choices, [], set())
        options = []
        m = self._copy_matrix()
        for perm in perms:
            options.extend(self._get_sequences(perm, 0, m, []))
        best = [True, 0]
        best_opt = options[0]
        done_so_far = 0
        
        # apply each Sequence
        for option in options:
            tmp_combo, broke_combo, rc_arr = self._get_placed_state(option, m)
            if not broke_combo or best[0]:
                res = self._assess_state(m, tmp_combo)
                if res > best[1]:
                    best[1] = res
                    best_opt = option
            self._undo_placed_state(option, m, rc_arr)
            done_so_far += 1
            # if done_so_far % 50 == 0:
            print(f'{done_so_far} / {len(options)} done so far')
            
        return best_opt

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
            r, c = self._remove_rows_and_cols(self.matrix)
            self._combo = self._adjust_combo(self._combo, r, c)
            
    def print_state(self):
        if all(c == 0 for c in self.choices):
            self._get_next_3()
        self._print_matrix()
        self._print_line()
        self._print_blocks()
