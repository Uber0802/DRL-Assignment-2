# n_tuple_network.py

import random
import math
import pickle
from pathlib import Path
from collections import namedtuple

import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import math

def board_to_exponents(board):
    """
    將 4x4 (0,2,4,8,...) -> 長度16 (exponent)
    0 -> 0, 2 ->1, 4->2, 8->3, ...
    """
    exps = []
    for v in board.flatten():
        if v == 0:
            exps.append(0)
        else:
            exps.append(int(math.log2(v)))
    return exps

def simulate_move(board, action, env_ref):
    """
    回傳「對 board 執行 action(只做合併)」後的新盤面 (不更新 score，不加 random tile)。
    可參考 is_move_legal() / move_up() 等邏輯，但不改 env_ref.score。
    """
    size = env_ref.size
    new_board = board.copy()

    def simulate_row_left(row):
        # 跟 environment simulate_row_move 一樣，但保證不更新score
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, size - len(new_row)), mode='constant')
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, size - len(new_row)), mode='constant')
        return new_row

    if action == 0:  # up
        for j in range(size):
            col = new_board[:, j]
            new_col = simulate_row_left(col)
            new_board[:, j] = new_col
    elif action == 1:  # down
        for j in range(size):
            col = new_board[:, j][::-1]
            new_col = simulate_row_left(col)
            new_board[:, j] = new_col[::-1]
    elif action == 2:  # left
        for i in range(size):
            row = new_board[i]
            new_board[i] = simulate_row_left(row)
    elif action == 3:  # right
        for i in range(size):
            row = new_board[i][::-1]
            new_row = simulate_row_left(row)
            new_board[i] = new_row[::-1]
    else:
        raise ValueError("Invalid action")

    return new_board


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        return not np.array_equal(self.board, temp_board)

class IllegalAction(Exception):
    pass


class nTupleNetwork:
    def __init__(self, tuples):
        self.TUPLES = tuples
        self.TARGET_PO2 = 15  # exponent base
        self.LUTS = self._init_LUTS()

    def _init_LUTS(self):
        LUTS = []
        for tp in self.TUPLES:
            size = (self.TARGET_PO2 + 1) ** len(tp)
            LUTS.append(np.zeros(size, dtype=np.float32))
        return LUTS

    def _tuple_id(self, values):
        n, k = 0, 1
        # reversed: 先處理最後一個 tile
        for v in reversed(values):
            if v >= self.TARGET_PO2:
                # 大於等於base就直接不允許
                raise ValueError("exponent >= base")
            n += v * k
            k *= self.TARGET_PO2
        return n

    def V(self, board_exps, delta=None):
        """返回該盤面價值 (n個 tuple LUT 平均)；若給 delta 就疊加更新"""
        vals = []
        for tp, LUT in zip(self.TUPLES, self.LUTS):
            tiles = [board_exps[i] for i in tp]
            idx = self._tuple_id(tiles)
            if delta is not None:
                LUT[idx] += delta
            vals.append(LUT[idx])
        return float(np.mean(vals))

    def evaluate(self, s, a, env_ref):
        """
        嘗試對 s 做 action a (只做 merge, 不 spawn)：
        若不可動 -> raise IllegalAction
        回傳 (r + V(after_state))；如果 < 0 就截為 0
        """
        # 建立環境複製
        board_arr = np.array([2**e if e>0 else 0 for e in s]).reshape(4, 4)
        env_copy = Game2048Env()
        env_copy.board = board_arr.copy()
        env_copy.score = 0

        # 做一次 move
        moved = False
        if a == 0:   moved = env_copy.move_up()
        elif a == 1: moved = env_copy.move_down()
        elif a == 2: moved = env_copy.move_left()
        elif a == 3: moved = env_copy.move_right()

        if not moved:
            raise IllegalAction()

        r = env_copy.score
        s_after = board_to_exponents(env_copy.board)
        value = r + self.V(s_after)
        # 若負則截斷
        return value if value > 0 else 0

    def best_action(self, s, env_ref):
        """
        回傳對 state s 具有最高 evaluate(s,a) 的 action。
        若都不能動就 raise IllegalAction
        """
        best_a = None
        best_val = -1
        for a in [0,1,2,3]:
            try:
                val = self.evaluate(s, a, env_ref)
                if val > best_val:
                    best_val = val
                    best_a = a
            except IllegalAction:
                pass
        if best_a is None:
            # 代表所有動作都 illegal
            raise IllegalAction()
        return best_a

    def learn(self, s, a, r, s_after, s_next, env_ref, alpha=0.01):
        """
        After-State TD(0): 
          v(s_after) <- v(s_after) + alpha * ( r_next + v(s_after_next) - v(s_after) )
        若所有 a_next 都 illegal，就當作 r_next=0, v_after_next=0
        """
        v_after = self.V(s_after)  # before update

        # 嘗試算出下一步 best action (在 s_next 上)
        try:
            a_next = self.best_action(s_next, env_ref)
            # 在 s_next 上只合併一次
            board_arr = np.array([2**e if e>0 else 0 for e in s_next]).reshape(4,4)
            env_copy = Game2048Env()
            env_copy.board = board_arr.copy()
            env_copy.score = 0

            moved = False
            if a_next == 0:  moved = env_copy.move_up()
            if a_next == 1:  moved = env_copy.move_down()
            if a_next == 2:  moved = env_copy.move_left()
            if a_next == 3:  moved = env_copy.move_right()

            if moved:
                r_next = env_copy.score
                s_after_next = board_to_exponents(env_copy.board)
                v_after_next = self.V(s_after_next)
            else:
                # 不該發生，但預防一下
                r_next = 0
                v_after_next = 0
        except IllegalAction:
            # s_next 上無法再動
            r_next = 0
            v_after_next = 0

        delta = r_next + v_after_next - v_after
        self.V(s_after, delta * alpha)
