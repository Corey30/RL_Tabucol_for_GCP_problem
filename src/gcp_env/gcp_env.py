import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
import tabucol
from PIL import Image
from gcp_env.ACO import AntColony


class GcpEnv(gym.Env):
    metadata = {"render_modes": ["human", "file"], "render_fps": 60}

    def __init__(
        self,
        graph,
        k,
        tabucol_iters=10000,
        beta=0.2,
        render_mode=None,
        base_filename=None,
        tabucol_init=False,#暂时不会在构造环境的时候跑一次（初始解）
        aco_init=False,
        aco_num_ants=15,
        aco_max_iter=30,
        disable_tabucol=False
    ):
        self._graph = graph
        self._k = k
        self.k = k
        self._adj_matrix = nx.to_numpy_array(graph, dtype=np.int32)
        self._disable_tabucol = disable_tabucol

        n = len(self._graph)

        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=0, high=n, shape=(n, 3), dtype=np.float32
                ),
                "col_features": spaces.Box(
                    low=0, high=n, shape=(n, k, 3), dtype=np.float32
                ),
                "k": spaces.Discrete(k, start=1),
            }
        )

        self.solution_space = spaces.Box(low=0, high=k - 1, shape=(n,), dtype=np.int32)

        self.action_space = spaces.MultiDiscrete([n, k])
        self._n = n

        self.color_map = None
        self.layout = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        if render_mode == "file":
            assert base_filename is not None
            self.render_iter = 0

        self._step_counter = 0

        adj_list = [[] for i in range(0, len(graph))]
        for node in graph:
            for neighbor in graph.neighbors(node):
                adj_list[node].append(neighbor)

        if not self._disable_tabucol:
            self._tabucol = tabucol.TabuColSolver(
                adj_list,
                k,
                max_iterations=tabucol_iters,
                tabu_a=10,
                tabu_alpha=1.2,
                beta=beta,
            )
        else:
            self._tabucol = None
        
        self._aco_init = aco_init
        self._aco_num_ants = aco_num_ants
        self._aco_max_iter = aco_max_iter
        self._aco = None 
        
        # 统计信息
        self._initial_score = 0
        self._rl_improvement = 0
        self._tabucol_improvement = 0
        self._action_stats = {
            'total_actions': 0,
            'effective_actions': 0,
            'node_selections': np.zeros(n),
            'color_selections': np.zeros(k)
        }#分析 agent 行为
        
        self.graph = graph
        self.final_score = 0
        self.final_solution = None
        self.episode_initial_score = 0
        self.episode_rl_improvement = 0
        self.episode_tabucol_improvement = 0
        
        self.last_episode_initial_score = 0
        self.last_episode_final_score = 0
        self.last_episode_rl_improvement = 0
        self.last_episode_tabucol_improvement = 0
        self.last_episode_solution = None

        #tabucol_init or aco_init or random_init
        if tabucol_init and not self._disable_tabucol:
            tabucol_sol, _ = self._tabucol.solve()
            self._solution = tabucol_sol
            self._prev_solution = tabucol_sol
        elif aco_init:
            print(f"[ACO] Initializing with ACO (ants={aco_num_ants}, iter={aco_max_iter})...")
            self._aco = AntColony(graph, k)
            aco_sol = self._aco.run(m=aco_num_ants, max_iter=aco_max_iter, verbose=True)
            self._solution = aco_sol
            self._prev_solution = aco_sol
            print(f"[ACO] Initialization complete, conflicts={self._aco.best_conflicts}")
        else:
            self._prev_solution = None
            self._solution = None

        self.render_mode = render_mode
        self.base_filename = base_filename

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.solution_space = spaces.Box(
            low=0, high=self._k - 1, shape=(self._n,), dtype=np.int32, seed=seed
        )
        self.action_space = spaces.MultiDiscrete([self._n, self._k], seed=seed)

        if options is not None and "initial_solution" in options:
            self._solution = options["initial_solution"]
            self._prev_solution = self._solution

        elif options is not None and options.get("use_aco", False):
          if self._aco is None:
            self._aco = AntColony(self._graph, self._k)
          print(f"[ACO] Starting initialization (ants={self._aco_num_ants}, iter={self._aco_max_iter})...")
          aco_sol = self._aco.run(m=self._aco_num_ants, max_iter=self._aco_max_iter, verbose=True)
          self._solution = aco_sol
          self._prev_solution = aco_sol
          print(f"[ACO] Initialization complete, conflicts={self._aco.best_conflicts}")
        elif options is not None and options.get("random_init", False):
            self._solution = self.solution_space.sample()
            self._prev_solution = self._solution
        #正常扰动
        elif self._prev_solution is not None:
            self._solution = self._prev_solution
        else:
            self._solution = self.solution_space.sample()
            self._prev_solution = self._solution

        self._step_counter = 0
        self._initialize_obs()#初始化观测特征
        self.score = self._calculate_score()
        
        #reset statistics
        self._initial_score = self.score
        self._rl_improvement = 0
        self._tabucol_improvement = 0
        self.episode_initial_score = 0
        self.episode_rl_improvement = 0
        self.episode_tabucol_improvement = 0
        self.final_score = 0
        self.final_solution = None
        if not hasattr(self, 'last_episode_initial_score'):
            self.last_episode_initial_score = 0
            self.last_episode_final_score = 0
            self.last_episode_rl_improvement = 0
            self.last_episode_tabucol_improvement = 0
            self.last_episode_solution = None
        self._action_stats = {
            'total_actions': 0,
            'effective_actions': 0,
            'node_selections': np.zeros(self._n),
            'color_selections': np.zeros(self._k)
        }
        
        observation = self._get_obs()

        return observation, {"solution": self._solution}

    def step(self, action):
        action_node, new_group = action[0], action[1]

        prev_group = self._solution[action_node]
        reward = 0
        
        self._action_stats['total_actions'] += 1
        self._action_stats['node_selections'][action_node] += 1
        self._action_stats['color_selections'][new_group] += 1

        if prev_group != new_group:
            self._action_stats['effective_actions'] += 1
            conflicts = self._node_feats[action_node, 1]

            self._solution[action_node] = new_group# 正式step

            new_conflicts = 0
            neighbors_groups = set()
            for neighbor in self._graph.neighbors(action_node):
                if self._solution[action_node] == self._solution[neighbor]:
                    new_conflicts += 1

                neighbors_groups.add(self._solution[neighbor])
                self._col_feats[neighbor, prev_group, 0] -= 1
                self._col_feats[neighbor, new_group, 0] += 1

                if self._solution[neighbor] == prev_group:
                    self._node_feats[neighbor, 1] -= 1
                if self._solution[neighbor] == new_group:
                    self._node_feats[neighbor, 1] += 1
                if self._col_feats[neighbor, prev_group, 0] == 0:
                    self._node_feats[neighbor, 2] -= 1 #如果邻居关于旧颜色的计数变成 0，就减少一个颜色组计数
                if self._col_feats[neighbor, new_group, 0] == 1:
                    self._node_feats[neighbor, 2] += 1#如果邻居关于新颜色的计数变成 1，就增加一个颜色组计数
            
            #更新action_node的特征值
            self._node_feats[action_node, 1] = new_conflicts #更新 action_node 自己的新冲突数
            self._node_feats[action_node, 2] = len(neighbors_groups)

            self._col_feats[:, prev_group, 1] -= 1#更新全局颜色组大小特征：旧颜色组大小 -1
            self._col_feats[:, new_group, 1] += 1 #更新全局颜色组大小特征：新颜色组大小 +1

            reward = conflicts - new_conflicts #及时奖励
            self._rl_improvement += reward

        self.score -= reward #更新全局 score

        terminated = False
        if self.score == 0:
            terminated = True

        self._step_counter += 1
        observation = self._get_obs()

        tabucol_rew = 0 #绝大多数 step，TabuCol 奖励就是 0
        if self._step_counter == self.spec.max_episode_steps:#如果当前已经走到 episode 最后一步，就调用 TabuCol
            if not self._disable_tabucol:
                score_before_tabucol = self.score
                self._tabucol.set_solution(self._solution)
                self._prev_solution, tabucol_rew = self._tabucol.solve()
                self._solution = self._prev_solution
                self._prev_solution = self._solution
                self._tabucol_improvement = score_before_tabucol - self._calculate_score()
                self.score = self._calculate_score()#再次更新全局score

        if terminated or self._step_counter == self.spec.max_episode_steps:
            self.final_score = self.score
            self.final_solution = self._solution.copy()
            self.episode_initial_score = self._initial_score
            self.episode_rl_improvement = self._rl_improvement
            self.episode_tabucol_improvement = self._tabucol_improvement
            self.last_episode_initial_score = self._initial_score
            self.last_episode_final_score = self.score
            self.last_episode_rl_improvement = self._rl_improvement
            self.last_episode_tabucol_improvement = self._tabucol_improvement
            self.last_episode_solution = self._solution.copy()

        info = {
            "solution": self._solution,
            "graph": self._graph,
            "initial_score": self._initial_score,
            "final_score": self.score,
            "rl_improvement": self._rl_improvement,
            "tabucol_improvement": self._tabucol_improvement,
        }

        return (
            observation,
            reward + tabucol_rew,
            terminated,
            False,
            info,
        )

    def render(self):
            # clear_output(wait=True)

        import matplotlib.pyplot as plt

        if self.color_map is None:
            self.color_map = dict(
                [
                    (
                        j,
                        f"#{''.join([random.choice('0123456789ABCDEF') for i in range(6)])}",
                    )
                    for j in range(self._k)
                ]
            )

        if self.layout is None:
            self.layout = nx.spring_layout(self._graph)

        fig, ax = plt.subplots(figsize=(10, 10))
        fig.tight_layout()
        ax.axis("off")
        node_colors = [self.color_map[node] for node in self._solution]

        edge_colors = [
            "#ff0000" if self._solution[x] == self._solution[y] else "#000000"
            for x, y in nx.edges(self._graph)
        ]
        alphas = [
            1.0 if self._solution[x] == self._solution[y] else 0.1
            for x, y in nx.edges(self._graph)
        ]

        plt.cla()
        nx.draw_networkx_nodes(self._graph, self.layout, node_color=node_colors)
        nx.draw_networkx_labels(self._graph, self.layout)
        nx.draw_networkx_edges(
            self._graph, self.layout, edge_color=edge_colors, alpha=alphas
        )
        fig.canvas.draw()

        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )

        if self.render_mode == "file":
            img = Image.fromarray(image_from_plot)
            print("saving", f"{self.base_filename}{self.render_iter}.png")
            img.save(f"{self.base_filename}{self.render_iter}.png")
            self.render_iter += 1
            plt.close()

        elif self.render_mode == "human":
            plt.show()

    def _get_obs(self):
        return {
            "node_features": self._node_feats,
            "col_features": self._col_feats,
            "k": self._k,
        }

    def _initialize_obs(self):
        # Features of node: (degree, conflicting edges, neighboring groups)
        node_feats = np.zeros((self._n, 3), dtype=np.float32)
        # Features of node, group pairs (i, j): (number of nodes in group j adjacent to i, total number of nodes in group j, |V|)
        col_feats = np.zeros((self._n, self._k, 3), dtype=np.float32)

        for node in self._graph.nodes():
            conflicts = 0
            groups = set()
            for adj in self._graph.neighbors(node):
                if self._solution[node] == self._solution[adj]:
                    conflicts += 1
                groups.add(self._solution[adj])
            node_feats[node, :] = [self._graph.degree(node), conflicts, len(groups)]

        nodes_with_color = np.zeros((self._k), dtype=np.int32)
        for col in self._solution:
            nodes_with_color[col] += 1

        for node in self._graph.nodes():
            for col in range(0, self._k):
                n_adjacent = 0
                for adj in self._graph.neighbors(node):
                    if self._solution[adj] == col:
                        n_adjacent += 1
                col_feats[node, col, :] = [n_adjacent, nodes_with_color[col], self._n]

        self._node_feats = node_feats
        self._col_feats = col_feats

    def _calculate_score(self):
        score = 0
        for node in self._graph.nodes():
            for neighbor in self._graph.neighbors(node):
                if self._solution[node] == self._solution[neighbor]:
                    score += 1

        score = score // 2
        return score

    def get_graph(self): 
        return self._graph

    def get_solution(self):
        return self._prev_solution
