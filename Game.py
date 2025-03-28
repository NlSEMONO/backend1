"""
Contains methods to help simulate block blast and judge block blast moves
"""
from numba import njit, prange
import numba
import numpy as np
import random

print(numba.set_num_threads(8))

R_BOUND = 8
L_BOUND = 0
PADDING = 100
all_blocks = 3
BLOCK_LEN = 9
ALL_CHOICES = 0

ONES = np.ones(R_BOUND * R_BOUND)

def _normalize(lst: list[int]):
    return lst + [PADDING for _ in range(BLOCK_LEN - len(lst))]

def _normalize_np_arr(arr: np.ndarray):
    return np.append(arr, [PADDING for _ in range(BLOCK_LEN - arr.shape[0])])

blocks = np.array([
    _normalize([]), # padding
    _normalize([8 * j + i for i in range(3) for j in range(3)]), # 3x3
    _normalize([8 * j + i for i in range(2) for j in range(2)]), # 2x2
])

@njit
def _adjust_combo(combo: int, r: np.ndarray, c: np.ndarray):
    return 3 if r.any() or c.any() else max(0, combo - 1)

def _add_rotations(block: np.ndarray, r: int) -> None:
    """Adds the first r rotations of block to self.blocks

    Args:
        block (np.ndarray): the block to be rotated
        r (int): number of 90 rotations to add to blocks list, starting at 0 degrees
    """
    globals()['blocks'] = np.append(blocks, [block], axis = 0)
    tmp = block[block < PADDING]
    for _ in range(r - 1):
        n = np.max(tmp // 8)
        tmp2 = 8 * (tmp % 8) + n - (tmp // 8)
        tmp = tmp2
        globals()['blocks'] = np.append(blocks, [_normalize_np_arr(tmp)], axis=0)

@njit
def _remove_block(b: int, x: int, y: int, matrix: np.ndarray) -> None:
    """
    Args:
        b (int): block index inside self.blocks
        x (int): top left y-coordinate placement location of the block
        y (int): top left x-coordinate placement location of the block
        matrix (np.ndarray): the matrix to remove the block from
    """
    for j in blocks[b][blocks[b] < PADDING]:
        matrix[j // 8 + x][j % 8 + y] = False

@njit
def _place_block(b: int, x: int, y: int, matrix: np.ndarray) -> None:
    """
    Args:
        b (int): block index to be placed
        x (int): top left y-coordinate placement location of the block
        y (int): top left x-coordinate placement location of the block
        matrix (np.ndarray): the matrix to place the block into
    """
    for j in blocks[b][blocks[b] < PADDING]:
        matrix[j // 8 + x][j % 8 + y] = True

@njit
def _validate_action(b: int, x: int, y: int, matrix: np.ndarray, print_error = True) -> bool:
    """
    Args:
        b (int): block index to be placed
        x (int): top left y-coordinate placement location of the block
        y (int): top left x-coordinate placement location of the block
        matrix (np.ndarray): matrix to place the block into
        print_error (bool, optional): Print errors? Defaults to True

    Returns:
        bool: Valid to place block at (y, x) or not?
    """
    if x < L_BOUND or x >= R_BOUND or y < L_BOUND or y >= R_BOUND:
        if (print_error):
            print("Invalid coordinates.")
            return False

    for j in blocks[b][blocks[b] < PADDING]:
        if j // 8 + x >= R_BOUND or j % 8 + y >= R_BOUND:
            return False
        elif matrix[j // 8 + x][j % 8 + y]:
            return False

    return True

@njit
def _remove_rows_and_cols(matrix: np.ndarray) -> list[np.ndarray]:
    """Removes completed rows and columns from matrix

    Args:
        matrix (np.ndarray): matrix to remove rows and columns from

    Returns:
        list[list[int]]: the rows and columns that got removed
    """
    # track rows and cols removed
    r = np.array([np.all(matrix[i] == True) for i in range(R_BOUND)])
    c = np.array([np.all(matrix[:,i] == True) for i in range(R_BOUND)])

    # reset rows and cols
    for i in range(R_BOUND):
        if r[i]:
            matrix[i] &= False

    for i in range(R_BOUND):
        if c[i]:
            matrix[:,i] &= False

    return [r, c]

@njit
def _add_rows_and_cols(r: list[int], c: list[int] , m: np.ndarray) -> None:
    """Opposite operation of _remove_rows_and_cols

    Args:
        r (list[int]): rows to add back
        c (list[int]): columns add back
        m (np.ndarray): matrix to add rows/cols back into
    """
    for i in range(R_BOUND):
        if r[i]:
            m[i] |= True
    for i in range(R_BOUND):
        if c[i]:
            m[:,i] |= True
        
@njit
def _validate_choice_wkr(blks: list[int], m: np.ndarray, idx: int) -> bool:
    if idx == 3:
        return True
    for i in range(R_BOUND):
        for j in range(R_BOUND):
            if _validate_action(blks[idx], i, j, m, False):
                _place_block(blks[idx], i, j, m)
                r, c = _remove_rows_and_cols(m)
                _validate_choice_wkr(blks, m, idx + 1)
                _add_rows_and_cols(r, c, m)
                _remove_block(blks[idx], i, j, m)
                return True
    return False

@njit
def _gen_perms(nums: list[int], perm_so_far: list[int], used: set[int], idx: int) -> set[tuple[int, int, int]]:
    """
    Args:
        nums (list[int]): tuple[int, int, int] as a list
        perm_so_far (list[int]): Accumulator variable for permutation so far
        used (set[int]): Set for tracking which numbers have already been used

    Returns:
        set[tuple[int, int, int]]: All possible orders to play the tuple[int, int, int]
    """
    res = set()
    # valid permutation, add result
    if idx == 3:
        res.add((perm_so_far[0], perm_so_far[1], perm_so_far[2]))
        return res
    # recurse
    n = len(nums)
    for i in range(n):
        if i not in used:
            used.add(i)
            perm_so_far[idx] = nums[i]
            res = res.union(_gen_perms(nums, perm_so_far, used, idx + 1))
            perm_so_far[idx] = -1
            used.remove(i)
    return res

@njit
def _validate_choice(blks: list[int], m: np.ndarray) -> bool:
    """
    Args:
        blks (list[int]): tuple[int, int, int] to validate
        m (np.ndarray): matrix to perform validation in

    Returns:
        bool: Does there exist a sequence of moves to get through the tuple[int, int, int]?
    """
    perms = _gen_perms(blks, [-1, -1, -1], {76, 69}, 0)
    for perm in perms:
        if _validate_choice_wkr(perm, m, 0):
            return True
    return False

@njit
def _gen_liveable_choices(matrix: np.ndarray) -> set[tuple[int, int, int]]:
    res = set()
    recur_matrix = _copy_matrix(matrix)

    # see if there is a placement of the permutation that does not result in death
    for i in range(1, all_blocks):
        for j in range(i, all_blocks):
            for k in range(j, all_blocks):
                if _validate_choice([i, j, k], recur_matrix):
                    res.add((i, j, k))
    return res

@njit
def _get_sequences(perm: tuple[int, int, int], idx: int, m: np.ndarray, seq_so_far: list[tuple[int, int, int]]) -> list[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]]:
    if idx == len(perm):
        return [(seq_so_far[0], seq_so_far[1], seq_so_far[2])]
    options = []
    for i in range(R_BOUND):
        for j in range(R_BOUND):
            if _validate_action(perm[idx], i, j, m, False):
                _place_block(perm[idx], i, j, m)
                r, c = _remove_rows_and_cols(m)
                seq_so_far[idx] = (perm[idx], i, j)
                options.extend(_get_sequences(perm, idx + 1, m, seq_so_far))
                seq_so_far[idx] = (-1, -1, -1)
                _add_rows_and_cols(r, c, m)
                _remove_block(perm[idx], i, j, m)
    return options

@njit
def _copy_matrix(matrix: np.ndarray):
    return np.array([[False for _ in range(8)] for _ in range(8)]) | matrix

@njit
def _get_placed_state(seq: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]], matrix: np.ndarray, combo: int) -> tuple[int, bool, list[tuple[list[int], list[int]]]]:
    broke_combo = False
    rc_arr = []
    for move in seq:
        _place_block(move[0], move[1], move[2], matrix)
        r, c = _remove_rows_and_cols(matrix)
        combo = _adjust_combo(combo, r, c)
        broke_combo = broke_combo or combo == 0
        rc_arr.append((r, c))
    return (combo, broke_combo, rc_arr)

@njit
def _undo_placed_state(seq: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]], matrix: np.ndarray, rc_arr: list[tuple[list[int], list[int]]]) -> None:
    for i in range(len(seq) - 1, -1, -1):
        _add_rows_and_cols(rc_arr[i][0], rc_arr[i][1], matrix)
        _remove_block(seq[i][0], seq[i][1], seq[i][2], matrix)

@njit
def _combo_maintained(matrix: np.ndarray, combo: int, perm: tuple[int, int, int], idx: int):
    if idx == 3:
        return True
    for i in range(R_BOUND):
        for j in range(R_BOUND):
            tmp_combo = combo
            if _validate_action(perm[idx], i, j, matrix, False):
                _place_block(perm[idx], i, j, matrix)
                r, c = _remove_rows_and_cols(matrix)
                tmp_combo = _adjust_combo(tmp_combo, r, c)
                if tmp_combo > 0:
                    if _combo_maintained(matrix, tmp_combo, perm, idx + 1):
                        return True
                _add_rows_and_cols(r, c, matrix)
                _remove_block(perm[idx], i, j, matrix)

    return False

@njit(parallel=True)
def _assess_state(matrix: np.ndarray, combo: int):
    no_break = 0
    res = list(_gen_liveable_choices(matrix))
    n = len(res)
    for i in prange(n):
        tmp_matrix = _copy_matrix(matrix)
        perms = _gen_perms(list(res[i]), [-1, -1, -1], {60, 59}, 0)
        for perm in perms:
            if _combo_maintained(tmp_matrix, combo, perm, 0):
                no_break += 1
                break
    return (no_break / len(res), len(res) / ALL_CHOICES)

@njit
def _print_matrix(matrix: np.ndarray, move) -> None:
    c = 0
    tmp = np.zeros(9) + blocks[move[0]] + 8 * move[1] + move[2]
    for i in range(R_BOUND):
        print_str = ""
        for j in range(R_BOUND):
            if (tmp == 8 * i + j).any():
                print_str += "@ "
            elif matrix[i][j]:
                print_str += "# "
            else:
                print_str += ". "
        print(f"{c + 1} {print_str}")
        c += 1
    print_str = " "
    for i in range(R_BOUND):
        print_str += f" {i + 1}"
    print(print_str)

def print_all_blocks():
    m = np.max((blocks % PADDING) // 8) + 1
    size = 6
    for i in range(m):
        print_str = ""
        for c in range(1, all_blocks):
            if c > 0 and np.max((blocks[c] % PADDING) // 8) >= i: 
                width = np.max(blocks[c][blocks[c] < PADDING] % 8) + 1
                buf = size - width
                print_str += " " * (buf // 2)
                for j in range(width):
                    print_str += '#' if any(blocks[c] == 8 * i + j) else '.'
                print_str += " " * ((buf // 2) + (buf % 2)) + "|"
            else:
                print_str += " " * (size) + "|"
        print(print_str)
    print_str = ""
    for i in range(all_blocks - 1):
        buf = size - len(str(i)) + 1
        print_str += " " * (buf // 2) + f"{i + 1}" + " " * ((buf // 2) + (buf % 2))
    print(print_str)

class Game:
    """Methods to run a game of block blast
    """
    matrix = np.array([[False for _ in range(8)] for _ in range(8)])
    _is_done = False
    true_symbol='#'
    false_symbol='.'
    choices = [0, 0, 0]
    _combo = 0
    perms_2 = np.array([
        _normalize([i for i in range(5)]), # 1x5
        _normalize([i for i in range(4)]), # 1x4
        _normalize([i for i in range(3)]), # 1x3
        _normalize([i for i in range(2)]), # 1x2
        _normalize([0,9,18]), # staircase 3x3
        _normalize([0,9]), # staircase 2x2
        _normalize([0,8,9,17]), # Z-shape 0
        _normalize([1,8,9,16]), # Z-shape reflect 0
        _normalize([8 * j + i for i in range(3) for j in range(2)]), # 2x3
    ], np.uint8)
    perms_4 = np.array([
        _normalize([0,1,2,9]), # T-shape 0
        _normalize([0,8,16,17]), # L-shape 0
        _normalize([1,9,16,17]), # L-shape reflect 0
        _normalize([0,8,16,17,18]), # L-shape big 0
        _normalize([0,8,9]), # L-shape small 0
    ], np.uint8)

    def __init__(self):
        for block in self.perms_2:
            _add_rotations(block, 2)

        for block in self.perms_4:
            _add_rotations(block, 4)
        
        globals()['all_blocks'] = blocks.shape[0]
        globals()['ALL_CHOICES'] = len(_gen_liveable_choices(_to_matrix(0)))

    def _print_matrix(self) -> None:
        c = 0
        for row in self.matrix:
            print(f"{c + 1} " + ' '.join(self.true_symbol if cell else self.false_symbol for cell in row))
            c += 1
        print(" ", end="")
        for i in range(R_BOUND):
            print(f" {i + 1}", end="")
        print()

    def _print_line(self) -> None:
        print()

    def _print_blocks(self) -> None:
        m = np.max((blocks[self.choices] % PADDING) // 8) + 1
        size = 6
        for i in range(m):
            print_str = ""
            for c in self.choices:
                if c > 0 and np.max((blocks[c] % PADDING) // 8) >= i: 
                    width = np.max(blocks[c][blocks[c] < PADDING] % 8) + 1
                    buf = size - width
                    print_str += " " * (buf // 2)
                    for j in range(width):
                        print_str += self.true_symbol if any(blocks[c] == 8 * i + j) else self.false_symbol
                    print_str += " " * ((buf // 2) + (buf % 2)) + "|"
                else:
                    print_str += " " * (size) + "|"
            print(print_str)
        print_str = ""
        for i in range(len(self.choices)):
            buf = size
            print_str += " " * (buf // 2) + f"{i + 1}" + " " * ((buf // 2) + (buf % 2))
        print(print_str)

    def is_done(self) -> bool:
        """
        Returns:
            bool: Is the game done or not?
        """
        return self._is_done
  
    def _get_next_3(self):
        res = list(_gen_liveable_choices(self.matrix))
        print(len(res))
        choice = random.choice(res)
        self.choices = list(choice)

    def _get_best_turn(self):
        perms = _gen_perms(self.choices, [-1, -1, -1], {69, 78}, 0)
        options = []
        m = _copy_matrix(self.matrix)
        for perm in perms:
            options.extend(_get_sequences(perm, 0, m, [(-1, -1, -1) for _ in range(3)]))
        best = [True, 0]
        best_opt = options[0]
        done_so_far = 0

        # apply each sequence
        for option in options:
            tmp_combo, broke_combo, rc_arr = _get_placed_state(option, m, self._combo)
            if not broke_combo or best[0]:
                res = _assess_state(m, tmp_combo)
                if res > best[1]:
                    best[1] = res
                    best_opt = option
            _undo_placed_state(option, m, rc_arr)
            done_so_far += 1
            if done_so_far % 50 == 0:
                print(f'{done_so_far} / {len(options)} done so far')

        return best_opt

    def perform_action(self, b: int, x: int, y: int):
        b -= 1
        x -= 1
        y -= 1
        if b >= 3 or b < 0:
            print("Invalid block selected")
            return
  
        if _validate_action(self.choices[b], x, y, self.matrix):
            _place_block(self.choices[b], x, y, self.matrix)
            self.choices[b] = 0
            r, c = _remove_rows_and_cols(self.matrix)
            self._combo = _adjust_combo(self._combo, r, c)
   
    def print_state(self):
        if all(c == 0 for c in self.choices):
            from datetime import datetime
            a = datetime.now()
            self._get_next_3()
            b = datetime.now()
            print(b-a)
        self._print_matrix()
        self._print_line()
        self._print_blocks()

@njit(locals={'x':numba.u8, 'i':numba.u8, 'j':numba.u8, 'num': numba.u8})
def _to_matrix(x: int):
    res = [[False for _ in range(R_BOUND)] for _ in range(R_BOUND)]
    num = 0
    for i in range(R_BOUND-1, -1, -1):
        for j in range(R_BOUND-1, -1, -1):
            num += 2**(8*i + j)
            if num <= x:
                res[i][j] = True
                x -= num
            num = 0
    return np.array(res)

@njit(locals={'x':numba.u8, 'i':numba.u8, 'j':numba.u8, 'num': numba.u8})
def _to_matrix1d(x: int):
    res = [False for _ in range(R_BOUND * R_BOUND)]
    num = 0
    for i in range(R_BOUND-1, -1, -1):
        for j in range(R_BOUND-1, -1, -1):
            num += 2**(8*i + j)
            if num <= x:
                res[8 * i + j] = True
                x -= num
            num = 0
    return np.array(res)

@njit(locals={'i':numba.u8, 'j':numba.u8, 'res':numba.uint64})
def _to_num(matrix: np.ndarray):
    res = 0
    for i in range(R_BOUND):
        for j in range(R_BOUND):
            if matrix[i][j]:
                res += 2**(8*i + j)
    return res

@njit
def _arr_to_num(arr: np.ndarray):
    res = 0
    for i in range(R_BOUND):
        if arr[i]:
            res += 2**i
    return res

@njit
def _num_to_arr(x: int):
    res = [False for _ in range(R_BOUND)]
    for i in range(R_BOUND-1, -1, -1):
        if 2**i <= x:
            res[i] = True
            x -= 2**i
    return np.array(res)

@njit(locals={'r':numba.types.bool[:], 'c':numba.types.bool[:],'combo':numba.u1})
def _unique_paths(matrix: np.ndarray, until: int, init_combo: int):
	# dp algorithm to avoid duping paths that have same combo at the same matrix id
    NUMBA_HINT_K, NUMBA_HINT_V = (int(2**63+1), 100), (100, 100, 100)
    RC_V = (0, 0)
    dp = [{NUMBA_HINT_K : NUMBA_HINT_V} for _ in range(4)]
    dp_rc = [{ NUMBA_HINT_K : RC_V } for _ in range(4)]
    dp[0][(_to_num(matrix), init_combo)] = (0, 100, 100)
    for i in range(4):
        del dp[i][NUMBA_HINT_K]
    for i in range(until):
        done_so_far = 0
        for m_id, combo in dp[i]:
            prev_m = _to_matrix(m_id)
            for j in range(1, all_blocks):
                for x in range(R_BOUND):
                    for y in range(R_BOUND):
                        if _validate_action(j, x, y, prev_m):
                            tmp_m = _copy_matrix(prev_m)
                            _place_block(j, x, y, tmp_m)
                            r, c = _remove_rows_and_cols(tmp_m)
                            next_key = (_to_num(tmp_m), _adjust_combo(combo, r, c))
                            dp[i+1][next_key] = (j, x, y)
                            dp_rc[i+1][next_key] = (_arr_to_num(r), _arr_to_num(c))
            done_so_far += 1
            if done_so_far % 3000 == 0:
                print(f'{done_so_far} / {len(dp[i])} :)))')
        print(f'{i} done {len(dp[i+1])}')
    return (dp, dp_rc)

# given 3 blocks, compute all future states
@njit(locals={'r':numba.types.bool[:], 'c':numba.types.bool[:],'combo':numba.u1,'m_id':numba.u8})
def _possible_futures(matrix: np.ndarray, choices: list[int], init_combo: int, must_combo: bool):
    # dp algorithm to avoid duping paths that have same combo at the same matrix id
    NUMBA_HINT_K, NUMBA_HINT_V = (int(2**63+1), 100), (100, 100, 100)
    RC_V = (0, 0, 0)
    dp = [{NUMBA_HINT_K : NUMBA_HINT_V} for _ in range(4)]
    dp_rc = [{ NUMBA_HINT_K : RC_V } for _ in range(4)]
    dp[0][(_to_num(matrix), init_combo)] = (0, 100, 100)
    perms = {(69, 69, 69), (69, 69, 69)}
    perms = _gen_perms(choices, [-1, -1, -1], {69, 42}, 0)
    for i in range(4):
        del dp[i][NUMBA_HINT_K]
    for i in range(3):
        done_so_far = 0
        for m_id, combo in dp[i]:
            prev_m = _to_matrix(m_id)
            for perm in perms:
                for x in range(R_BOUND):
                    for y in range(R_BOUND):
                        if _validate_action(perm[i], x, y, prev_m):
                            tmp_m = _copy_matrix(prev_m)
                            _place_block(perm[i], x, y, tmp_m)
                            r, c = _remove_rows_and_cols(tmp_m)
                            next_key = (_to_num(tmp_m), _adjust_combo(combo, r, c))
                            if next_key[1] > 0 or not must_combo:
                                dp[i+1][next_key] = (perm[i], x, y)
                                dp_rc[i+1][next_key] = (_arr_to_num(r), _arr_to_num(c), combo)
            done_so_far += 1
            if done_so_far % 3000 == 0:
                print(f'{done_so_far} / {len(dp[i])} :)))')
        print(f'{i} done {len(dp[i+1])}')
    return (dp, dp_rc)

# from final states, choose the ones that maintain desired combo and the resultant matrix has >= 30 squares filled
@njit
def _get_good_states(pathfinder: list[dict[tuple[int, int], tuple[int, int, int, int]]], rc, choices: list[int]):
    perms = {(69, 69, 69), (69, 69, 69)}
    perms = _gen_perms(choices, [-1, -1, -1], {69, 42}, 0)
    print(perms)
    good_states = [(3, 64, int(2**63 + 1))]
    good_states.pop()
    TOO_MUCH_COMPUTE = 0
    while len(good_states) == 0 and TOO_MUCH_COMPUTE >= 0:
        for key in pathfinder[3]:
            if np.sum(ONES[_to_matrix1d(key[0])]) >= TOO_MUCH_COMPUTE:
                acc = []
                curr = key
                tmp_m = _to_matrix(key[0])
                for i in range(3, 0, -1):
                    r, c, cmb = rc[i][curr]
                    steps = pathfinder[i][curr]
                    _add_rows_and_cols(_num_to_arr(r), _num_to_arr(c), tmp_m)
                    _remove_block(steps[0], steps[1], steps[2], tmp_m)
                    acc.append(steps[0])
                    curr = (_to_num(tmp_m), cmb)
                if (acc[2], acc[1], acc[0]) in perms:
                    good_states.append((key[1], np.sum(ONES[_to_matrix1d(key[0])]), key[0]))
        TOO_MUCH_COMPUTE -= 1
    good_states.sort(reverse=True)
    print(f'Found {len(good_states)} good states')
    ret = [(int(2**63 + 1), 1)]
    ret.pop()
    ret = [(state[2], state[0]) for state in good_states]
    return ret

@njit
def _to_matrix_njit_wrapper(state):
    return _to_matrix(state[0])

@njit
def _apply_path(state, paths, curr_rc, matrix, init_combo):
    st = [(0, 0, 0)]
    st.pop()
    tmp_m = _to_matrix_njit_wrapper(state)
    curr = state
    for i in range(3, 0, -1):
        r, c, cmb = curr_rc[i][curr]
        steps = paths[i][curr]
        _add_rows_and_cols(_num_to_arr(r), _num_to_arr(c), tmp_m)
        _remove_block(steps[0], steps[1], steps[2], tmp_m)
        st.append((steps[0], steps[1], steps[2]))
        curr = (_to_num(tmp_m), cmb)
    _print_matrix(tmp_m, (0, 0, 0))
    print()
    while len(st) > 0:
        trans = st[-1]
        st.pop()
        _place_block(trans[0], trans[1], trans[2], matrix)
        r, c = _remove_rows_and_cols(matrix)
        init_combo = _adjust_combo(init_combo, r, c)
        _print_matrix(matrix, (trans[0], trans[1], trans[2]))
        print()
    return init_combo

@njit
def _test_next_state(state):
    return _assess_state(_to_matrix(state[0]), state[1])

@njit
def _check_good_states(good_states, paths, curr_rc, matrix, init_combo):
    best = (-1, -1, 0)
    best_combo = good_states[0][1]
    LIVE_BREAKPOINT = 1 - (1 / 500) # 1/500 chance of death
    # COMBO_SACRIFICE = (1 / 100) # sacrifice 1% better chance to combo for better chance to live
    for i in range(min(len(good_states), 100)):
        state = good_states[i]
        res_c, res_l = _test_next_state(state)
        if best[0] < res_c and state[1] == best_combo and (res_l >= LIVE_BREAKPOINT or res_l > best[1]):
            best = (res_c, res_l, i)
        # print(f'{i + 1} / {min(len(good_states), 100)}')
        # print(f"Use: {res}?")
        # inp = input()
        # if inp[0] == "y":
        #     return _apply_path(state, paths, curr_rc, matrix, init_combo)
        # elif inp[0] == "-" and int(inp) * -1 - 1 <= i:
        #     idx = int(inp) * -1
        #     return _apply_path(good_states[i-idx], paths, curr_rc, matrix, init_combo)
    combo_pct = best[0] * 10000
    live_pct = best[1] * 10000
    print(f"checked {min(len(good_states), 100)} good states, choosing {best[2] + 1}th option with combo maintain probability of {int(combo_pct // 100)}.{int(combo_pct%100)}% and chance to live of {int(live_pct // 100)}.{int(live_pct%100)}%")
    return _apply_path(good_states[best[2]], paths, curr_rc, matrix, init_combo)

@njit
def get_move(matrix: np.ndarray, choices: list[int], init_combo: int):
    # _unique_paths(matrix, 3, init_combo)
    paths, curr_rc = _possible_futures(matrix, choices, init_combo, True)
    good_states = [int(2**64-1)]
    good_states = _get_good_states(paths, curr_rc, choices)
    if len(good_states) > 0:
        return _check_good_states(good_states, paths, curr_rc, matrix, init_combo)
    else:
        print("BROKE COMBO :(")
        paths, curr_rc = _possible_futures(matrix, choices, init_combo, False)
        good_states = [int(2**64-1)]
        good_states = _get_good_states(paths, curr_rc, choices)
        return _check_good_states(good_states, paths, curr_rc, matrix, init_combo)
