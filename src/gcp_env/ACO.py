"""
标准ACO算法用于图着色问题 - 项目集成版本

直接替换原有的 AntColony 类即可使用
保持相同的接口，但使用标准ACO实现

对比原版本的改进：
1. 移除所有非标准机制（解池、动态调色、多样性注入）
2. 修正启发信息设计（基于冲突数而非颜色使用次数）
3. 简化信息素更新逻辑
4. 标准化参数设置
5. 提高稳定性和可预测性
"""

import networkx as nx
import numpy as np


class AntColony:
    """
    标准蚁群算法求解图着色问题
    
    专为warm-start场景优化：
    - 快速收敛（20-30迭代）
    - 稳定输出（参数保守）
    - 简洁实现（无复杂机制）
    """
    
    def __init__(self, graph, max_colors):
        """
        初始化ACO求解器
        
        Args:
            graph: NetworkX图对象
            max_colors: 允许使用的最大颜色数（如果为None则设为节点数）
        """
        self.graph = graph
        self.max_colors = max_colors if max_colors else len(graph)
        self.num_vertices = len(graph)
        
        # 邻接矩阵和邻接表
        self.adj_matrix = nx.to_numpy_array(graph)
        self.adj_list = [np.where(row)[0].tolist() for row in self.adj_matrix]
        
        # 节点度数
        self.degrees = np.sum(self.adj_matrix, axis=1)
        
        # === ACO核心参数（标准配置）===
        self.alpha = 1.0      # 信息素重要性
        self.beta = 2.0       # 启发信息重要性  
        self.rho = 0.5        # 挥发系数（从0.08改为0.5，标准值）
        self.Q = 100.0        # 信息素强度常数
        
        # 信息素矩阵：均匀初始化（标准做法）
        tau_0 = 0.1
        self.Tau = np.ones((self.num_vertices, self.max_colors)) * tau_0
        
        # 记录最优解
        self.best_solution = None
        self.best_conflicts = float('inf')
        
        # 节点处理顺序：按度数降序
        self.vertex_order = np.argsort(-self.degrees)
        
        # 用于绘图的历史记录（保持接口兼容）
        self.iter_x = []
        self.iter_y = []
    
    def calculate_conflicts(self, solution):
        """
        计算解的冲突数
        
        Args:
            solution: 着色方案 np.ndarray
        
        Returns:
            int: 冲突边数
        """
        conflicts = 0
        for v in range(self.num_vertices):
            for u in self.adj_list[v]:
                if u > v and solution[v] == solution[u]:
                    conflicts += 1
        return conflicts
    
    def calculate_heuristic(self, vertex, color, current_solution):
        """
        计算启发信息 η(vertex, color)
        
        核心改进：基于该颜色会产生的冲突数
        η = 1 / (1 + 冲突数)
        
        Args:
            vertex: 当前节点
            color: 候选颜色
            current_solution: 当前部分着色方案
        
        Returns:
            float: 启发信息值
        """
        conflicts = 0
        for neighbor in self.adj_list[vertex]:
            if current_solution[neighbor] == color:
                conflicts += 1
        
        return 1.0 / (1.0 + conflicts)
    
    def generate_ant_solution(self):
        """
        单只蚂蚁构建完整解
        
        改进点：
        1. 移除随机的local_alpha/local_beta
        2. 启发信息改为基于冲突数
        3. 简化节点处理顺序选择
        
        Returns:
            np.ndarray: 着色方案
        """
        solution = np.full(self.num_vertices, -1, dtype=np.int32)
        
        # 节点处理顺序：固定按度数降序 + 小幅随机扰动
        processing_order = self.vertex_order.copy()
        
        # 小幅随机化（保持探索能力）
        if self.num_vertices > 10:
            swap_count = 2  # 固定交换2对
            for _ in range(swap_count):
                i, j = np.random.choice(self.num_vertices, 2, replace=False)
                processing_order[i], processing_order[j] = processing_order[j], processing_order[i]
        
        # 逐节点着色
        for v in processing_order:
            neighbors = self.adj_list[v]
            used_colors = {solution[u] for u in neighbors if solution[u] != -1}
            
            # 移除-1（未着色标记）
            used_colors.discard(-1)
            
            # 可用颜色
            available_colors = [c for c in range(self.max_colors) if c not in used_colors]
            
            if available_colors:
                # 有无冲突的颜色可选
                probs = np.zeros(len(available_colors), dtype=np.float64)
                
                for i, c in enumerate(available_colors):
                    # 标准ACO公式：P ∝ τ^α × η^β
                    tau = self.Tau[v, c]
                    eta = self.calculate_heuristic(v, c, solution)
                    probs[i] = (tau ** self.alpha) * (eta ** self.beta)
                
                # 归一化
                prob_sum = probs.sum()
                if prob_sum > 0:
                    probs /= prob_sum
                else:
                    probs = np.ones(len(available_colors)) / len(available_colors)
                
                # 轮盘赌选择
                try:
                    chosen_color = np.random.choice(available_colors, p=probs)
                except:
                    chosen_color = np.random.choice(available_colors)
            
            else:
                # 所有颜色都会冲突，选择冲突最少的
                conflict_counts = np.zeros(self.max_colors)
                for neighbor in neighbors:
                    if solution[neighbor] != -1:
                        conflict_counts[solution[neighbor]] += 1
                
                min_conflicts = conflict_counts.min()
                min_conflict_colors = np.where(conflict_counts == min_conflicts)[0]
                
                # 从最少冲突的颜色中，根据信息素选择
                if len(min_conflict_colors) > 1:
                    probs = self.Tau[v, min_conflict_colors] ** self.alpha
                    probs /= (probs.sum() + 1e-10)
                    try:
                        chosen_color = np.random.choice(min_conflict_colors, p=probs)
                    except:
                        chosen_color = np.random.choice(min_conflict_colors)
                else:
                    chosen_color = min_conflict_colors[0]
            
            solution[v] = chosen_color
        
        return solution
    
    def light_local_improvement(self, solution):
        """
        简单局部搜索（保留原有接口）
        
        改进：简化逻辑，提高效率
        
        Args:
            solution: 输入解
        
        Returns:
            np.ndarray: 改进后的解
        """
        improved = solution.copy()
        
        # 找出冲突节点
        conflict_vertices = []
        for v in range(self.num_vertices):
            current_color = improved[v]
            has_conflict = any(improved[neighbor] == current_color 
                             for neighbor in self.adj_list[v])
            if has_conflict:
                conflict_vertices.append(v)
        
        # 随机打乱
        np.random.shuffle(conflict_vertices)
        
        # 限制处理数量（warm-start不需要完美）
        process_limit = min(len(conflict_vertices), max(10, len(conflict_vertices) // 5))
        
        for v in conflict_vertices[:process_limit]:
            current_color = improved[v]
            current_conflicts = sum(1 for neighbor in self.adj_list[v]
                                  if improved[neighbor] == current_color)
            
            if current_conflicts == 0:
                continue
            
            # 找冲突最少的颜色
            best_color = current_color
            best_conflicts = current_conflicts
            
            for c in range(self.max_colors):
                if c != current_color:
                    new_conflicts = sum(1 for neighbor in self.adj_list[v]
                                      if improved[neighbor] == c)
                    
                    if new_conflicts < best_conflicts:
                        best_conflicts = new_conflicts
                        best_color = c
            
            # 接受改进
            if best_conflicts < current_conflicts:
                improved[v] = best_color
        
        return improved
    
    def update_pheromones(self, solutions):
        """
        更新信息素矩阵
        
        改进：简化为标准ACO更新
        
        Args:
            solutions: 所有蚂蚁的解列表
        """
        # 计算所有解的冲突数
        conflicts_list = [self.calculate_conflicts(sol) for sol in solutions]
        
        # 更新全局最优
        min_idx = np.argmin(conflicts_list)
        if conflicts_list[min_idx] < self.best_conflicts:
            self.best_conflicts = conflicts_list[min_idx]
            self.best_solution = solutions[min_idx].copy()
        
        # 步骤1: 信息素挥发
        self.Tau *= (1 - self.rho)
        
        # 步骤2: 所有蚂蚁增强信息素
        for sol, conflicts in zip(solutions, conflicts_list):
            # 质量越好，增量越大
            delta = self.Q / (conflicts + 1.0)
            
            for v in range(self.num_vertices):
                c = sol[v]
                if 0 <= c < self.max_colors:
                    self.Tau[v, c] += delta
        
        # 限制信息素范围
        self.Tau = np.clip(self.Tau, 0.01, 10.0)
    
    def run(self, m=None, max_iter=30, verbose=False, enable_dynamic_colors=False):
        """
        运行ACO算法（保持原接口兼容）
        
        Args:
            m: 蚂蚁数量（默认15，轻量级）
            max_iter: 最大迭代次数（默认30）
            verbose: 是否打印信息（默认False）
            enable_dynamic_colors: 忽略（不再使用动态调色）
        
        Returns:
            np.ndarray: 着色方案，shape=(num_vertices,), dtype=np.int32
        """
        num_ants = 15 if m is None else m
        
        if verbose:
            print(f"开始ACO优化：{self.num_vertices}节点，{self.max_colors}颜色")
            print(f"参数：m={num_ants}, iter={max_iter}, α={self.alpha}, β={self.beta}, ρ={self.rho}")
        
        for iteration in range(max_iter):
            solutions = []
            
            # 所有蚂蚁构建解
            for _ in range(num_ants):
                ant_solution = self.generate_ant_solution()
                
                # 25%概率进行局部搜索
                if np.random.random() < 0.25:
                    ant_solution = self.light_local_improvement(ant_solution)
                
                solutions.append(ant_solution)
            
            # 更新信息素
            self.update_pheromones(solutions)
            
            # 记录历史（用于绘图）
            self.iter_x.append(iteration)
            self.iter_y.append(self.best_conflicts)
            
            # 定期输出
            if verbose and iteration % 10 == 0:
                print(f"迭代{iteration}: 最优冲突={self.best_conflicts}")
            
            # 早停
            if self.best_conflicts == 0:
                if verbose:
                    print(f"在迭代{iteration}找到无冲突解")
                break
        
        if verbose:
            print(f"ACO完成: 最终冲突={self.best_conflicts}")
        
        # 确保返回正确格式
        if self.best_solution is None:
            self.best_solution = np.random.randint(0, self.max_colors, 
                                                  self.num_vertices, dtype=np.int32)
        
        # 确保值域正确
        result = np.asarray(self.best_solution, dtype=np.int32)
        result = np.clip(result, 0, self.max_colors - 1)
        
        return result


# ===============================
# 测试代码
# ===============================

if __name__ == "__main__":
    import time
    
    print("="*70)
    print("标准ACO算法测试 - 项目集成版本")
    print("="*70)
    
    # 测试1: 小图
    print("\n【测试1】小图（50节点，p=0.3）")
    graph = nx.gnp_random_graph(50, 0.3, seed=42)
    aco = AntColony(graph, max_colors=10)
    
    start = time.time()
    solution = aco.run(m=15, max_iter=30, verbose=True)
    elapsed = time.time() - start
    
    print(f"\n结果:")
    print(f"  运行时间: {elapsed:.2f}秒")
    print(f"  解的形状: {solution.shape}")
    print(f"  解的类型: {solution.dtype}")
    print(f"  值域: [{solution.min()}, {solution.max()}]")
    print(f"  冲突数: {aco.calculate_conflicts(solution)}")
    
    # 测试2: 中等规模图（模拟实际使用场景）
    print("\n" + "="*70)
    print("【测试2】中等规模图（250节点，p=0.5）- 模拟项目场景")
    graph = nx.gnp_random_graph(250, 0.5, seed=42)
    aco = AntColony(graph, max_colors=24)
    
    start = time.time()
    solution = aco.run(m=15, max_iter=30, verbose=False)
    elapsed = time.time() - start
    
    print(f"结果:")
    print(f"  运行时间: {elapsed:.2f}秒")
    print(f"  冲突数: {aco.calculate_conflicts(solution)}")
    print(f"  实际使用颜色: {len(set(solution))}")
    
    # 对比随机初始化
    print("\n对比随机初始化:")
    random_sol = np.random.randint(0, 24, 250)
    random_conflicts = aco.calculate_conflicts(random_sol)
    print(f"  随机解冲突数: {random_conflicts}")
    print(f"  ACO解冲突数: {aco.calculate_conflicts(solution)}")
    improvement = (random_conflicts - aco.calculate_conflicts(solution)) / random_conflicts * 100
    print(f"  改进: {improvement:.1f}%")
    
    print("\n" + "="*70)
    print("测试完成！可直接集成到项目中使用")
    print("="*70)