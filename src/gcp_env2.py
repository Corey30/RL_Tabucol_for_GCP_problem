"""
GcpEnv2 - Paper-style Reactive Tabu Search Environment for Graph Coloring

Enhanced version with 4D state and scaled reward:
- Observation: 4D compact state (delta_f_scaled, h_norm, q_norm, t_norm)
- Action: 25 T_f levels, direct tenure assignment
- Reward: scaled (best_before - best_in_epoch) / num_edges + weak success bonus
- Epoch-based control: RL only intervenes at epoch boundaries
"""

import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
from typing import List, Optional, Tuple

src_path = os.path.dirname(os.path.abspath(__file__))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

tabucol_path = os.path.join(src_path, 'tabucol', 'python')
if tabucol_path not in sys.path:
    sys.path.insert(0, tabucol_path)

import tabucol


class GcpEnv2(gym.Env):
    """
    Paper-style Reactive Tabu Search Environment for Graph Coloring (Enhanced 4D State)

    Paper Mapping:
    - variables x_i -> graph nodes
    - clauses C_j -> graph edges
    - f(x) = #unsatisfied clauses -> f(x) = #conflict edges
    - n = #variables -> n = #nodes
    - m = #clauses -> m = #edges
    - Hamming distance H -> coloring Hamming distance
    - prohibition parameter T -> tabu tenure
    - fractional T_f = T/n -> fractional T_f = tenure/n

    Observation (Enhanced 4D):
        delta_f_scaled = clip(scale_df * (f_epoch - best_score_before_epoch) / num_edges, -5, 5)
        h_norm = H_epoch / num_nodes
        q_norm = log(1 + current_score) / log(1 + initial_score)  [0, 1]
        t_norm = (T_f - 0.01) / (0.25 - 0.01)  [0, 1]

    Action (paper-style 25 levels):
        action in {0, 1, ..., 24}
        T_f = (action + 1) * 0.01  -> T_f in {0.01, 0.02, ..., 0.25}
        tenure = floor(num_nodes * T_f)

    Reward (Enhanced with scale):
        r = clip(scale * (best_score_before_epoch - best_in_epoch) / num_edges, -2, 2)
        + success_bonus * 0.1 if score == 0
    """

    def __init__(
        self,
        graph,
        k: int,
        epoch_length: Optional[int] = None,
        max_epochs: int = 100,
        success_reward: float = 1.0,
        seed: Optional[int] = None,
        reward_scale: float = 300.0,
        delta_f_scale: float = 1000.0,
    ):
        super().__init__()
        
        self._graph = graph
        self._k = k
        self._n = len(graph)
        self._num_edges = len(list(graph.edges()))

        if epoch_length is None:
            self._epoch_length = max(int(self._n * 0.5), 500)
        else:
            self._epoch_length = epoch_length

        self._max_epochs = max_epochs
        self._success_reward = success_reward

        self._reward_scale = reward_scale
        self._delta_f_scale = delta_f_scale

        self._T_f_values = np.array([(i+1) * 0.01 for i in range(25)], dtype=np.float32)
        self._current_T_f = 0.01
        
        self.observation_space = spaces.Box(
            low=np.array([-5.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([5.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(25)

        self._adj_list = self._build_adj_list()

        #定义Tabu搜索求解器
        self._solver = tabucol.TabuColSolver(
            graph=self._adj_list,
            k=self._k,
            max_iterations=0,
            tabu_a=10,
            beta=0.2,
            seed=seed,
        )

        self._current_solution: np.ndarray = np.zeros(self._n, dtype=np.int32)
        self._current_score: int = 0
        self._best_score_ever: int = 0
        self._initial_score: int = 0
        self._epoch_counter: int = 0
        self._episode_seed: int = 0

    def _build_adj_list(self) -> List[List[int]]:
        adj_list = [[] for _ in range(self._n)]
        for u, v in self._graph.edges():
            adj_list[u].append(v)
            adj_list[v].append(u)
        return adj_list

    def _calculate_score(self, solution: np.ndarray) -> int:
        score = 0
        for u, v in self._graph.edges():
            if solution[u] == solution[v]:
                score += 1
        return score

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._episode_seed = seed if seed is not None else 42
        np.random.seed(self._episode_seed)

        if options is not None and "initial_solution" in options:
            self._current_solution = np.array(options["initial_solution"], dtype=np.int32)
        else:
            self._current_solution = np.random.randint(0, self._k, size=self._n, dtype=np.int32)

        self._initial_score = self._calculate_score(self._current_solution)
        self._current_score = self._initial_score
        self._best_score_ever = self._initial_score
        self._epoch_counter = 0
        self._current_T_f = 0.01
        
        self._solver.set_seed(self._episode_seed)
        self._solver.set_solution(self._current_solution.tolist())
        
        obs = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        info = {}

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        T_f = self._T_f_values[action]
        self._current_T_f = T_f
        tenure = max(1, int(self._n * T_f))

        best_before_epoch = self._best_score_ever
        score_start_epoch = self._current_score
        solution_at_epoch_start = self._current_solution.copy()

        self._solver.set_solution(self._current_solution.tolist())
        self._solver.set_tabu_a(tenure)

        end_solution, end_score, summary = self._solver.run_epoch(
            self._epoch_length,
            solution_at_epoch_start.tolist(),
        )
        
        self._current_solution = np.array(end_solution, dtype=np.int32)
        self._current_score = end_score
        
        f_epoch = summary.f_epoch
        best_in_epoch = summary.best_in_epoch
        H_epoch = summary.h_epoch

        if best_in_epoch < self._best_score_ever:
            self._best_score_ever = best_in_epoch
        
        delta_f = (f_epoch - best_before_epoch) / max(1, self._num_edges)
        delta_f_scaled = np.clip(self._delta_f_scale * delta_f, -5.0, 5.0)
        
        h_norm = H_epoch / max(1, self._n)
        h_norm_clipped = np.clip(h_norm, 0.0, 1.0)
        
        if self._initial_score > 0:
            q_norm = np.log(1 + end_score) / np.log(1 + self._initial_score)
        else:
            q_norm = 1.0
        q_norm_clipped = np.clip(q_norm, 0.0, 1.0)
        
        t_norm = (T_f - 0.01) / (0.25 - 0.01)
        t_norm_clipped = np.clip(t_norm, 0.0, 1.0)
        
        obs = np.array([delta_f_scaled, h_norm_clipped, q_norm_clipped, t_norm_clipped], dtype=np.float32)

        reward_raw = (score_start_epoch - f_epoch) / max(1, self._num_edges)
        reward_scaled = np.clip(100.0 * reward_raw, -2.0, 2.0)
        reward = reward_scaled

        self._epoch_counter += 1

        terminated = (end_score == 0)
        truncated = (self._epoch_counter >= self._max_epochs)

        info = {
            "epoch": self._epoch_counter,
            "score": end_score,
            "best_score_ever": self._best_score_ever,
            "best_before_epoch": best_before_epoch,
            "best_in_epoch": best_in_epoch,
            "f_epoch": f_epoch,
            "H_epoch": H_epoch,
            "T_f": float(T_f),
            "tenure": tenure,
            "moves_executed": summary.moves_executed,
            "reward_raw": reward_raw,
            "reward_scaled": reward_scaled,
            "success": (end_score == 0),
            "delta_f": float(delta_f),
            "delta_f_scaled": float(delta_f_scaled),
            "h_norm": float(h_norm_clipped),
            "q_norm": float(q_norm_clipped),
            "t_norm": float(t_norm_clipped),
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        print(f"Epoch: {self._epoch_counter}, Score: {self._current_score}, "
              f"Best: {self._best_score_ever}")

    def close(self):
        pass


def make_env(
    n_nodes: int = 250,
    edge_prob: float = 0.5,
    n_colors: int = 24,
    epoch_length: Optional[int] = None,
    max_epochs: int = 100,
    seed: Optional[int] = None,
    reward_scale: float = 300.0,
    delta_f_scale: float = 1000.0,
):
    graph = nx.gnp_random_graph(n_nodes, edge_prob, seed=seed)
    return GcpEnv2(
        graph=graph,
        k=n_colors,
        epoch_length=epoch_length,
        max_epochs=max_epochs,
        seed=seed,
        reward_scale=reward_scale,
        delta_f_scale=delta_f_scale,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("GcpEnv2 Enhanced 4D State Environment Test")
    print("=" * 60)

    env = make_env(n_nodes=100, n_colors=10, epoch_length=500, max_epochs=20)
    obs, info = env.reset(seed=42)

    print(f"\nInitial Info: {info}")
    print(f"Initial Observation: {obs}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    print(f"  - delta_f_scaled: {obs[0]:.4f}")
    print(f"  - h_norm:         {obs[1]:.4f}")
    print(f"  - q_norm:         {obs[2]:.4f}")
    print(f"  - t_norm:         {obs[3]:.4f}")

    print("\nRunning 5 epochs with random actions...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Epoch {i+1}: action={action}, T_f={info['T_f']:.2f}, "
              f"tenure={info['tenure']}, score={info['score']}, "
              f"best_in_epoch={info['best_in_epoch']}, reward={reward:.4f}")
        print(f"           obs=[{obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f}, {obs[3]:.3f}]")
        if terminated or truncated:
            break

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
