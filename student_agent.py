# student_agent.py

import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import math
from n_tuple_network import nTupleNetwork, IllegalAction
import pickle
import MCTS
from MCTS import TD_MCTS_Node, TD_MCTS
from tqdm import trange





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


class NtupleApproximator:
    def __init__(self, n_tuple_agent):
        self.n_tuple_agent = n_tuple_agent

    def value(self, board):
        """
        回傳該盤面的評估值 (float)。
        這裡直接調用 n_tuple_agent.V(...)
        """
        exps = board_to_exponents(board)
        return self.n_tuple_agent.V(exps)


import gdown
from pathlib import Path


MODEL_DIR  = Path("models_try")
MODEL_PATH = MODEL_DIR / "nTupleNet_30000games.pkl"
# DRIVE_ID    = "1yQCKO8FqA85VFJGRRmGBbogK5CfSRFNf"
# DRIVE_ID   =  "14YbYyWUUINlI4LfarzUgOXK1e7VXYPG4"
DRIVE_ID   = "12-n-syQLTDkh2AMROM0gWtNl-10VZXIy"
URL        = f"https://drive.google.com/uc?id={DRIVE_ID}"

NTUPLE_AGENT = None
IS_LOADED = False
if not MODEL_PATH.exists():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pretrained model to {MODEL_PATH} …")
    gdown.download(URL, str(MODEL_PATH), quiet=False)
else:
    print(f"Found pretrained model at {MODEL_PATH}")

def load_ntuple_agent():
    global NTUPLE_AGENT, IS_LOADED
    import sys
    import n_tuple_network
    sys.modules["__main__"].nTupleNetwork = n_tuple_network.nTupleNetwork
    sys.modules["__main__"].IllegalAction = n_tuple_network.IllegalAction

    if not IS_LOADED:
        # checkpoint_path = "/tmp2/b11902127/DRL-Assignment-2/models_try/nTupleNet_30000games.pkl"
        with open(MODEL_PATH, "rb") as f:
            n_games, agent = pickle.load(f)
            NTUPLE_AGENT = agent
        IS_LOADED = True
        print(f"[INFO] Loaded nTupleNetwork from {MODEL_PATH}, trained {n_games} games.")


last_score = 0
def get_action(state, score):
    global last_score
    if score > last_score + 5000:
        last_score = score
        print("score checkpoint : ", last_score)
    
    load_ntuple_agent()
    approximator = NtupleApproximator(NTUPLE_AGENT)
    env_temp = Game2048Env()
    

    env_temp.board = state.copy()
    env_temp.score = score
    
    mcts = TD_MCTS(env=env_temp, approximator=approximator, iterations=500, exploration_constant=1.41, rollout_depth=8, gamma=0.99)
    root_node = TD_MCTS_Node(state=state, score=score, parent=None, action=None)

    for _ in range(mcts.iterations):
        mcts.run_simulation(root_node)

    best_a, dist = mcts.best_action_distribution(root_node)
    if best_a is None:
        return random.choice([0,1,2,3])
    return best_a