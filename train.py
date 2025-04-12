import random
import math
import pickle
from pathlib import Path
from collections import namedtuple

import numpy as np
import gym

from student_agent import Game2048Env 


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


class IllegalAction(Exception):
    pass

Transition = namedtuple("Transition", "s a r s_after s_next")
Gameplay   = namedtuple("Gameplay",   "transitions total_reward max_tile")

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


##########################
# 3) 訓練流程 (跟第二份相同)
##########################
def play_one_game(agent, env):
    """
    直接用 best_action 玩一局，直到沒合法動作或盤面滿。
    每次 merge 完後再手動 spawn 新tile
    """
    transitions = []
    total_r = 0

    env.reset()
    while True:
        # 先記錄 s
        s = board_to_exponents(env.board)

        # 由 agent 決定 best action
        try:
            a = agent.best_action(s, env)
        except IllegalAction:
            # 沒有任何合法動作 => 結束
            break

        # 執行 action (只做合併)
        prev_score = env.score
        moved = False
        if a == 0:   moved = env.move_up()
        elif a == 1: moved = env.move_down()
        elif a == 2: moved = env.move_left()
        elif a == 3: moved = env.move_right()
        if not moved:
            # 也視為非法 => 結束
            break
        r = env.score - prev_score
        total_r += r

        # after-state
        s_after = board_to_exponents(env.board)

        # spawn tile => next-state
        if not env.is_game_over():
            env.add_random_tile()
            s_next = board_to_exponents(env.board)
        else:
            # game over
            s_next = s_after  # 其實已經結束
            # 不過寫成一樣也行
            pass

        transitions.append(Transition(s, a, r, s_after, s_next))
        if env.is_game_over():
            break

    max_tile = int(env.board.max())
    return Gameplay(transitions, total_r, max_tile)

def learn_from_gameplay(agent, gameplay, env_ref, alpha=0.1):
    for tr in reversed(gameplay.transitions[:-1]):
        agent.learn(tr.s, tr.a, tr.r, tr.s_after, tr.s_next, env_ref, alpha=alpha)


if __name__ == "__main__":
    
    TUPLES = [
        [0, 1, 2, 4, 5, 6],
        [3, 7, 11, 2, 6, 10],
        [15, 14, 13, 11, 10, 9],
        [12, 8, 4, 13, 9, 5],
        [3, 2, 1, 7, 6, 5],
        [12, 13, 14, 8, 9, 10],
        [0, 4, 8, 1, 5, 9],
        [15, 11, 7, 14, 10, 6],
        [1, 2, 5, 6, 9, 13],
        [7, 11, 6, 10, 5, 4],
        [14, 13, 10, 9, 6, 2],
        [8, 4, 9, 5, 10, 11],
        [2, 1, 6, 5, 10, 14],
        [13, 14, 9, 10, 5, 1],
        [4, 8, 5, 9, 6, 7],
        [11, 7, 10, 6, 9, 8],
        [0, 1, 2, 3, 4, 5],
        [3, 7, 11, 15, 2, 6],
        [15, 14, 13, 12, 11, 10],
        [12, 8, 4, 0, 13, 9],
        [3, 2, 1, 0, 7, 6],
        [12, 13, 14, 15, 8, 9],
        [0, 4, 8, 12, 1, 5],
        [15, 11, 7, 3, 14, 10],
        [0, 1, 5, 6, 7, 10],
        [3, 7, 6, 10, 14, 9],
        [15, 14, 10, 9, 8, 5],
        [12, 8, 9, 5, 1, 6],
        [3, 2, 6, 5, 4, 9],
        [12, 13, 9, 10, 11, 6],
        [0, 4, 5, 9, 13, 10],
        [15, 11, 10, 6, 2, 5],
        [0, 1, 2, 5, 9, 10],
        [3, 7, 11, 6, 5, 9],
        [15, 14, 13, 10, 6, 5],
        [12, 8, 4, 9, 10, 6],
        [3, 2, 1, 6, 10, 9],
        [12, 13, 14, 9, 5, 6],
        [0, 4, 8, 5, 6, 10],
        [15, 11, 7, 10, 9, 5],
        [0, 1, 5, 9, 13, 14],
        [3, 7, 6, 5, 4, 8],
        [15, 14, 10, 6, 2, 1],
        [12, 8, 9, 10, 11, 7],
        [3, 2, 6, 10, 14, 13],
        [12, 13, 9, 5, 1, 2],
        [0, 4, 5, 6, 7, 11],
        [15, 11, 10, 9, 8, 4],
        [0, 1, 5, 8, 9, 13],
        [3, 7, 6, 1, 5, 4],
        [15, 14, 10, 7, 6, 2],
        [12, 8, 9, 14, 10, 11],
        [3, 2, 6, 11, 10, 14],
        [12, 13, 9, 4, 5, 1],
        [0, 4, 5, 2, 6, 7],
        [15, 11, 10, 13, 9, 8],
        [0, 1, 2, 4, 6, 10],
        [3, 7, 11, 2, 10, 9],
        [15, 14, 13, 11, 9, 5],
        [12, 8, 4, 13, 5, 6],
        [3, 2, 1, 7, 5, 9],
        [12, 13, 14, 8, 10, 6],
        [0, 4, 8, 1, 9, 10],
        [15, 11, 7, 14, 6, 5],
    ]

    env = Game2048Env()
    agent = nTupleNetwork(TUPLES)

    n_session = 5000   # 跟第二份例子一樣
    n_episode = 100    # 每個 session 做 100 局

    save_dir = Path("models_try")
    save_dir.mkdir(exist_ok=True)

    n_games = 0
    try:
        for i_se in range(n_session):
            gameplays = []
            for _ in range(n_episode):
                gp = play_one_game(agent, env)
                learn_from_gameplay(agent, gp, env, alpha=0.1)
                gameplays.append(gp)
                n_games += 1

            rewards = [g.total_reward for g in gameplays]
            maxtiles = [g.max_tile for g in gameplays]
            mean_reward = np.mean(rewards)
            mean_maxtile = np.mean(maxtiles)
            n2048 = sum(1 for mt in maxtiles if mt >= 2048)
            print(f"[Session {i_se+1}/{n_session}] Games: {n_games}, "
                  f"MeanReward: {mean_reward:.1f}, "
                  f"MeanMaxTile: {mean_maxtile:.1f}, "
                  f"2048-rate: {n2048/len(gameplays):.2%}, "
                  f"BestTile: {max(maxtiles)}")

            if (i_se+1) % 50 == 0:
                fn = save_dir / f"nTupleNet_{n_games}games.pkl"
                with open(fn, "wb") as f:
                    pickle.dump((n_games, agent), f)
                print("Model saved to", fn)

    except KeyboardInterrupt:
        print("Training interrupted after", n_games, "games")
        fn = save_dir / f"nTupleNet_{n_games}games_interrupt.pkl"
        with open(fn, "wb") as f:
            pickle.dump((n_games, agent), f)
        print("Agent saved to", fn)
