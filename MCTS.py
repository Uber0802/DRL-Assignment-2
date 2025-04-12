# MCTS.py

import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import math
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


class TD_MCTS_Node:
    def __init__(self, state, score, parent=None, action=None, is_random_node=False, exploration_constant=1.41):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        is_random_node: whether this node is BEFORE random tile placement
        exploration_constant: the exploration parameter (c) for UCT
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.is_random_node = is_random_node
        self.c = exploration_constant  # Store exploration constant in each node
        self.children = {}
        self.visits = 0
        self.total_value = 0.0
        self.untried_actions = []

        if not self.is_random_node:
            self.untried_actions = [a for a in range(4) if self.is_move_legal(a)]
        else:
            self.untried_actions = self.get_possible_tile_placements()

    def is_move_legal(self, action):
        temp_env = Game2048Env()
        temp_env.board = self.state.copy()
        temp_env.score = self.score
        return temp_env.is_move_legal(action)

    def get_possible_tile_placements(self):
        empty_cells = list(zip(*np.where(self.state == 0)))
        possible_placements = []
        for cell in empty_cells:
            possible_placements.append((cell, 2))  # 90% chance
            possible_placements.append((cell, 4))  # 10% chance
        return possible_placements

    def fully_expanded(self):
        return len(self.untried_actions) == 0

    def get_uct_value(self, parent_visits, v_norm=1.0):
        """
        UCT formula:
            UCT = (average_value / v_norm) + c * sqrt( log(parent_visits) / visits )

        v_norm can be dynamically computed by the parent to normalize exploitation.
        """
        if self.visits == 0:
            return float('inf')

        average_value = self.total_value / self.visits
        normalized_value = average_value / 9500
        exploration_term = self.c * math.sqrt(math.log(parent_visits) / self.visits)

        # print("Value1: ", normalized_value)
        # print("Value2: ", exploration_term)

        return normalized_value + exploration_term


class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=0, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        if node.is_random_node:
            placements = list(node.children.keys())
            weights = [0.9 if placement[1] == 2 else 0.1 for placement in placements]
            selected_placement = random.choices(placements, weights=weights)[0]
            return node.children[selected_placement]
        else:
        
            v_norm_candidates = []

            for child in node.children.values():
                if child.visits > 0:
                    avg_value = child.total_value / child.visits
                    v_norm_candidates.append(avg_value)

            if len(v_norm_candidates) == 0:
                # Fallback if no child visited
                v_norm = 1200
            else:
                v_norm = max(v_norm_candidates)

            best_child = None
            best_value = -float('inf')
            for child in node.children.values():
                uct_value = child.get_uct_value(node.visits, v_norm=v_norm)
                if uct_value > best_value:
                    best_value = uct_value
                    best_child = child

            return best_child

    def expand(self, node):
        if node.is_random_node:
            untried = node.untried_actions[:]

            # tile_placement = random.choice(untried)
            untried = node.untried_actions[:]
            weights = [0.9 if v == 2 else 0.1 for (_, v) in untried]
            tile_placement = random.choices(untried, weights=weights)[0]

            (x, y), value = tile_placement
            # print
            new_state = node.state.copy()
            new_state[x, y] = value

            is_duplicate = any(np.array_equal(child.state, new_state) for child in node.children.values())
            if is_duplicate:
                print("[WARNING] duplicate child detected!!")
                node.untried_actions.remove(tile_placement)
                # continue

            new_score = node.score
            child_node = TD_MCTS_Node(
                new_state, new_score,
                parent=node,
                action=None,
                is_random_node=False,
                exploration_constant=self.c
            )
            node.children[tile_placement] = child_node
            node.untried_actions.remove(tile_placement)

            return child_node

            # return None

        else:
            # Expand an action
            action = random.choice(node.untried_actions)
            sim_env = self.create_env_from_state(node.state, node.score)

            # Execute action without adding random tile
            if action == 0:
                sim_env.move_up()
            elif action == 1:
                sim_env.move_down()
            elif action == 2:
                sim_env.move_left()
            elif action == 3:
                sim_env.move_right()

            new_state = sim_env.board.copy()
            new_score = sim_env.score
            reward = new_score - node.score

            child_node = TD_MCTS_Node(
                new_state, new_score,
                parent=node,
                action=action,
                is_random_node=True,
                exploration_constant=self.c
            )
            child_node.reward = reward
            node.children[action] = child_node
            node.untried_actions.remove(action)

        return child_node

    def rollout(self, node, depth):
        total_reward = 0
        discount = 1.0
        sim_env = self.create_env_from_state(node.state, node.score)

        for _ in range(depth):
            if sim_env.is_game_over():
                break

            legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_actions:
                break
            action = random.choice(legal_actions)

            old_score = sim_env.score
            _, reward, done, _ = sim_env.step(action)
            actual_reward = reward - old_score
            total_reward += discount * actual_reward
            discount *= self.gamma

        legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
        if not legal_moves:
            return 0

        values = []
        pre_random_states = []
        for a in legal_moves:
            tmp_env = copy.deepcopy(sim_env)
            score_before = tmp_env.score
            if a == 0:
                tmp_env.move_up()
            elif a == 1:
                tmp_env.move_down()
            elif a == 2:
                tmp_env.move_left()
            elif a == 3:
                tmp_env.move_right()
            reward_sim = tmp_env.score - score_before
            s_candidate = tmp_env.board.copy()
            # print(approximator.value(s_candidate))
            value_est = reward_sim + self.gamma * self.approximator.value(s_candidate)
            values.append(value_est)
            pre_random_states.append(s_candidate)

        final_value = max(values)
        total_reward += discount * final_value

        return total_reward


    def backpropagate(self, node, reward):
        discount = 1.0
        while node is not None:
            node.visits += 1
            node.total_value += discount * reward
            if hasattr(node, 'reward'):
                reward += node.reward

            discount *= self.gamma
            node = node.parent



    def run_simulation(self, root):
        node = root

        # Selection
        d = 0
        while node.fully_expanded():
            if not node.children:
                break
            d += 1
            node = self.select_child(node)

        # print("Depth: ", d)

        # Expansion
        if not node.fully_expanded():
            node = self.expand(node)

        # Rollout
        if node.is_random_node:
            # For random nodes, use the approximator directly
            rollout_value = self.approximator.value(node.state)
        else:
            # For action nodes, perform rollout
            rollout_value = self.rollout(node, self.rollout_depth)

        # Backpropagation
        self.backpropagate(node, rollout_value)

    def best_action_distribution(self, root):
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None

        for action, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0

        return best_action, distribution

    def search(self, initial_state, initial_score):
        root = TD_MCTS_Node(
            initial_state, initial_score,
            is_random_node=False,
            exploration_constant=self.c
        )

        for _ in range(self.iterations):
            self.run_simulation(root)

        return self.best_action_distribution(root)
    


