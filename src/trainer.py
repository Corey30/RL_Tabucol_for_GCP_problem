import torch
from tianshou.utils.net.common import ActorCritic
from tianshou.data import VectorReplayBuffer, Collector
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.logger.base import BaseLogger
import gymnasium as gym
from network import ActorNetwork, CriticNetwork, GCPPPOPolicy
import networkx as nx
from gcp_env import GcpEnv
from tianshou.env import SubprocVectorEnv
import argparse
from gymnasium.envs.registration import register
from collections import defaultdict
import json
import os
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class TrainingMonitorLogger(BaseLogger):
    """自定义 logger，记录训练指标，不影响训练逻辑"""
    
    def __init__(self, save_dir="training_logs"):
        super().__init__(train_interval=1, test_interval=1, update_interval=1)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.metrics = defaultdict(list)
        self.epoch = 0
        self.best_test_reward = float('-inf')
        self.best_epoch = 0
        
    def write(self, step_type: str, step: int, data: dict):
        """记录数据"""
        for key, value in data.items():
            if isinstance(value, (np.integer, np.floating)):
                value = float(value)
            elif isinstance(value, np.ndarray):
                value = float(value.mean())
            self.metrics[key].append((step, value))
    
    def log_train_data(self, collect_result: dict, step: int) -> None:
        """记录训练数据"""
        if collect_result["n/ep"] > 0:
            log_data = {
                "train/reward": collect_result["rew"],
                "train/length": collect_result["len"],
            }
            self.write("train/env_step", step, log_data)
    
    def log_test_data(self, collect_result: dict, step: int) -> None:
        """记录测试数据"""
        assert collect_result["n/ep"] > 0
        log_data = {
            "test/reward": collect_result["rew"],
            "test/length": collect_result["len"],
            "test/reward_std": collect_result["rew_std"],
            "test/length_std": collect_result["len_std"],
        }
        self.write("test/env_step", step, log_data)
        
        if collect_result["rew"] > self.best_test_reward:
            self.best_test_reward = collect_result["rew"]
            self.best_epoch = self.epoch
    
    def log_update_data(self, update_result: dict, step: int) -> None:
        """记录更新数据"""
        log_data = {f"update/{k}": v for k, v in update_result.items()}
        self.write("update/gradient_step", step, log_data)
    
    def save_data(self, epoch, env_step, gradient_step, save_checkpoint_fn=None):
        self.epoch = epoch
        if save_checkpoint_fn:
            save_checkpoint_fn(epoch, env_step, gradient_step)
    
    def restore_data(self):
        return 0, 0, 0
    
    def record_epoch_metrics(self, epoch, metrics_dict):
        """记录每个 epoch 的详细指标"""
        for key, value in metrics_dict.items():
            if isinstance(value, (np.integer, np.floating)):
                value = float(value)
            elif isinstance(value, np.ndarray):
                value = float(value.mean())
            self.metrics[f"epoch/{key}"].append((epoch, value))
    
    def save_metrics(self):
        """保存指标到 JSON"""
        metrics_file = os.path.join(self.save_dir, "training_metrics.json")
        
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, tuple):
                return [convert(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        with open(metrics_file, 'w') as f:
            json.dump(convert(dict(self.metrics)), f, indent=2)
        print(f"Metrics saved to {metrics_file}")
    
    def plot_curves(self):
        """绘制训练曲线"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(3, 3, figsize=(18, 14))
            fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
            
            if 'epoch/train_score_mean' in self.metrics:
                epochs, values = zip(*self.metrics['epoch/train_score_mean'])
                axes[0, 0].plot(epochs, values, 'b-', label='Train', alpha=0.7, linewidth=2)
            if 'epoch/test_score_mean' in self.metrics:
                epochs, values = zip(*self.metrics['epoch/test_score_mean'])
                axes[0, 0].plot(epochs, values, 'r-', label='Test', alpha=0.7, linewidth=2)
                axes[0, 0].axhline(y=min(values), color='g', linestyle='--', 
                                   label=f'Best: {min(values):.1f}', linewidth=1.5)
            if 'epoch/train_score_mean' in self.metrics or 'epoch/test_score_mean' in self.metrics:
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Conflicts')
                axes[0, 0].set_title('Score (Conflicts) - Lower is Better')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            if 'train/reward' in self.metrics:
                steps, values = zip(*self.metrics['train/reward'])
                axes[0, 1].plot(steps, values, 'b-', alpha=0.7, linewidth=2)
            if 'test/reward' in self.metrics:
                steps, values = zip(*self.metrics['test/reward'])
                axes[0, 1].plot(steps, values, 'r-', alpha=0.7, linewidth=2)
                axes[0, 1].axhline(y=self.best_test_reward, color='g', linestyle='--', 
                                   label=f'Best: {self.best_test_reward:.1f}', linewidth=1.5)
            if 'train/reward' in self.metrics or 'test/reward' in self.metrics:
                axes[0, 1].set_xlabel('Env Step')
                axes[0, 1].set_ylabel('Reward')
                axes[0, 1].set_title('Reward - Higher is Better')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            if 'update/loss/clip' in self.metrics:
                steps, values = zip(*self.metrics['update/loss/clip'])
                axes[0, 2].plot(steps, values, 'orange', alpha=0.7, linewidth=2)
                axes[0, 2].set_xlabel('Gradient Step')
                axes[0, 2].set_ylabel('Loss')
                axes[0, 2].set_title('Policy Loss (Clip)')
                axes[0, 2].grid(True, alpha=0.3)
            
            if 'update/loss/vf' in self.metrics:
                steps, values = zip(*self.metrics['update/loss/vf'])
                axes[1, 0].plot(steps, values, 'purple', alpha=0.7, linewidth=2)
                axes[1, 0].set_xlabel('Gradient Step')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].set_title('Value Loss')
                axes[1, 0].grid(True, alpha=0.3)
            
            if 'update/loss/ent' in self.metrics:
                steps, values = zip(*self.metrics['update/loss/ent'])
                axes[1, 1].plot(steps, values, 'green', alpha=0.7, linewidth=2)
                axes[1, 1].set_xlabel('Gradient Step')
                axes[1, 1].set_ylabel('Entropy')
                axes[1, 1].set_title('Policy Entropy (Exploration)')
                axes[1, 1].grid(True, alpha=0.3)
            
            if 'epoch/perfect_rate' in self.metrics:
                epochs, values = zip(*self.metrics['epoch/perfect_rate'])
                axes[1, 2].plot(epochs, values, 'g-', linewidth=2)
                axes[1, 2].set_xlabel('Epoch')
                axes[1, 2].set_ylabel('Rate')
                axes[1, 2].set_title('Perfect Solution Rate')
                axes[1, 2].set_ylim([0, 1])
                axes[1, 2].grid(True, alpha=0.3)
            
            if 'epoch/rl_improvement' in self.metrics and 'epoch/tabucol_improvement' in self.metrics:
                epochs1, rl_values = zip(*self.metrics['epoch/rl_improvement'])
                epochs2, tabu_values = zip(*self.metrics['epoch/tabucol_improvement'])
                x = np.arange(len(epochs1))
                width = 0.35
                axes[2, 0].bar(x - width/2, rl_values, width, label='RL', alpha=0.7)
                axes[2, 0].bar(x + width/2, tabu_values, width, label='TabuCol', alpha=0.7)
                axes[2, 0].set_xlabel('Epoch')
                axes[2, 0].set_ylabel('Improvement')
                axes[2, 0].set_title('RL vs TabuCol Contribution')
                axes[2, 0].set_xticks(x[::max(1, len(x)//10)])
                axes[2, 0].set_xticklabels([epochs1[i] for i in range(0, len(epochs1), max(1, len(epochs1)//10))])
                axes[2, 0].legend()
                axes[2, 0].grid(True, alpha=0.3, axis='y')
            
            if 'epoch/improvement_ratio' in self.metrics:
                epochs, values = zip(*self.metrics['epoch/improvement_ratio'])
                axes[2, 1].plot(epochs, values, 'b-', linewidth=2)
                axes[2, 1].set_xlabel('Epoch')
                axes[2, 1].set_ylabel('Ratio')
                axes[2, 1].set_title('Improvement Ratio')
                axes[2, 1].grid(True, alpha=0.3)
            
            if 'epoch/initial_score_mean' in self.metrics:
                epochs, values = zip(*self.metrics['epoch/initial_score_mean'])
                axes[2, 2].plot(epochs, values, 'b-', label='Initial', alpha=0.7, linewidth=2)
            if 'epoch/test_score_mean' in self.metrics:
                epochs, values = zip(*self.metrics['epoch/test_score_mean'])
                axes[2, 2].plot(epochs, values, 'r-', label='Final', alpha=0.7, linewidth=2)
            if 'epoch/initial_score_mean' in self.metrics or 'epoch/test_score_mean' in self.metrics:
                axes[2, 2].set_xlabel('Epoch')
                axes[2, 2].set_ylabel('Score')
                axes[2, 2].set_title('Initial vs Final Score')
                axes[2, 2].legend()
                axes[2, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = os.path.join(self.save_dir, "training_curves.png")
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Training curves saved to {plot_file}")
        except Exception as e:
            print(f"Warning: Failed to generate plots: {e}")
    
    def print_summary(self):
        """打印训练摘要"""
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        print(f"Best Test Reward: {self.best_test_reward:.2f} (Epoch {self.best_epoch})")
        print(f"Total Epochs: {self.epoch}")
        
        if 'epoch/test_score_mean' in self.metrics:
            epochs, scores = zip(*self.metrics['epoch/test_score_mean'])
            print(f"Best Test Score: {min(scores):.1f} conflicts")
            print(f"Final Test Score: {scores[-1]:.1f} conflicts")
            print(f"Score Improvement: {scores[0]:.1f} -> {scores[-1]:.1f}")
        
        if 'epoch/perfect_rate' in self.metrics:
            epochs, rates = zip(*self.metrics['epoch/perfect_rate'])
            print(f"Best Perfect Solution Rate: {max(rates):.2%}")
            print(f"Final Perfect Solution Rate: {rates[-1]:.2%}")
        
        if 'update/loss/ent' in self.metrics:
            steps, entropies = zip(*self.metrics['update/loss/ent'])
            print(f"Final Entropy: {entropies[-1]:.4f}")
        
        print("="*70 + "\n")


def calculate_score(graph, solution):
    """计算冲突数"""
    if graph is None or solution is None:
        return -1
    score = 0
    for node in graph.nodes():
        for neighbor in graph.neighbors(node):
            if solution[node] == solution[neighbor]:
                score += 1
    return score // 2


def evaluate_policy_detailed(policy, test_collector, test_envs, num_episodes=10, verbose=False):
    """
    使用 Collector 进行评估，避免手动处理 obs 形状导致的张量维度不匹配问题。
    
    核心改动：
    - 不再手动构建 Batch、手动调用 policy()、手动解码 action
    - 直接用 Tianshou 的 Collector.collect() 完成数据收集
    - 从环境属性中读取 last_episode_* 获取详细的 RL/TabuCol 分阶段指标
    
    这样 Collector 内部会正确处理：
    1. SubprocVectorEnv 返回的 obs 格式和批量维度
    2. policy 的前向推理和 action 采样
    3. GCPPPOPolicy.map_action() 的 flat_action -> (node, col) 转换
    4. 环境的 step() 调用和 episode 终止判断
    """
    if verbose:
        print(f"  Starting detailed evaluation for {num_episodes} episodes...")
    
    # 重置 collector 状态，确保从干净状态开始
    test_collector.reset()
    test_collector.reset_env()
    test_collector.reset_buffer()
    
    # 用 Collector 收集指定数量的完整 episode
    # Collector 内部会正确处理所有 obs 格式转换和 action 映射
    collect_result = test_collector.collect(n_episode=num_episodes)
    
    if verbose:
        print(f"  Collector finished: {collect_result['n/ep']} episodes, "
              f"avg_reward={collect_result['rew']:.2f}, avg_len={collect_result['len']:.1f}")
    
    # 从各个子环境中读取 last_episode_* 属性
    # 这些属性在 GcpEnv.step() 的 episode 结束时已经被正确记录
    num_envs = len(test_envs)
    
    scores = []
    initial_scores = []
    rl_improvements = []
    tabucol_improvements = []
    
    for env_id in range(num_envs):
        try:
            final_score = test_envs.get_env_attr('last_episode_final_score', id=env_id)
            init_score = test_envs.get_env_attr('last_episode_initial_score', id=env_id)
            rl_imp = test_envs.get_env_attr('last_episode_rl_improvement', id=env_id)
            tabu_imp = test_envs.get_env_attr('last_episode_tabucol_improvement', id=env_id)
            
            # get_env_attr 返回列表，取第一个元素
            if isinstance(final_score, (list, np.ndarray)):
                final_score = final_score[0]
            if isinstance(init_score, (list, np.ndarray)):
                init_score = init_score[0]
            if isinstance(rl_imp, (list, np.ndarray)):
                rl_imp = rl_imp[0]
            if isinstance(tabu_imp, (list, np.ndarray)):
                tabu_imp = tabu_imp[0]
            
            # 只记录有效数据（last_episode_final_score 初始为 0，
            # 但如果 episode 真正跑完了，init_score 一般 > 0）
            if init_score is not None and init_score > 0:
                scores.append(max(0, int(final_score)))
                initial_scores.append(max(0, int(init_score)))
                rl_improvements.append(float(rl_imp) if rl_imp else 0.0)
                tabucol_improvements.append(float(tabu_imp) if tabu_imp else 0.0)
                
                if verbose:
                    rl_sign = "+" if rl_imp < 0 else "-"
                    rl_val = abs(rl_imp)
                    tabu_sign = "-" if tabu_imp > 0 else "+"
                    tabu_val = abs(tabu_imp)
                    print(f"    Env {env_id}: initial={init_score} -> "
                          f"RL({rl_sign}{rl_val:.0f}冲突) -> TabuCol({tabu_sign}{tabu_val:.0f}冲突) -> final={final_score}")
        except Exception as e:
            if verbose:
                print(f"    Env {env_id}: Error reading attributes - {e}")
    
    if not scores:
        if verbose:
            print("  Warning: No valid episode data collected")
        return None
    
    avg_initial = np.mean(initial_scores)
    avg_final = np.mean(scores)
    improvement_ratio = (avg_initial - avg_final) / max(avg_initial, 1) if avg_initial > 0 else 0
    
    result = {
        'test_score_mean': float(avg_final),
        'test_score_min': float(np.min(scores)),
        'test_score_std': float(np.std(scores)),
        'test_reward_mean': float(collect_result['rew']),
        'initial_score_mean': float(avg_initial),
        'rl_improvement': float(np.mean(rl_improvements)),
        'tabucol_improvement': float(np.mean(tabucol_improvements)),
        'improvement_ratio': float(improvement_ratio),
        'perfect_rate': float(sum(1 for s in scores if s == 0) / len(scores)),
    }
    
    if verbose:
        print(f"\n  Evaluation Summary:")
        print(f"    Episodes evaluated: {len(scores)}")
        print(f"    Initial Score: {result['initial_score_mean']:.1f}")
        print(f"    RL Improvement: {result['rl_improvement']:.1f}")
        print(f"    TabuCol Improvement: {result['tabucol_improvement']:.1f}")
        print(f"    Final Score: {result['test_score_mean']:.1f} (min={result['test_score_min']:.0f})")
        print(f"    Improvement Ratio: {result['improvement_ratio']:.2%}")
        print(f"    Perfect Rate: {result['perfect_rate']:.2%}")
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reinforcement learning based tabu search for graph coloring trainer"
    )
    parser.add_argument("output", type=str, help="Path to policy output file")
    parser.add_argument(
        "--input", type=str, default=None, help="Path to policy input file"
    )
    parser.add_argument(
        "-I",
        "--max-steps",
        type=int,
        dest="max_steps",
        default=300,
        help="Maximum number of steps per episode",
    )
    parser.add_argument(
        "-T",
        "--tabucol-iters",
        type=int,
        dest="tabucol_iters",
        default=5000,
        help="Number of iterations for tabucol",
    )
    parser.add_argument(
        "-E",
        "--epochs",
        type=int,
        dest="epochs",
        default=50,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "-N",
        "--nodes",
        type=int,
        dest="nodes",
        default=250,
        help="Number of nodes in training graphs",
    )
    parser.add_argument(
        "-P",
        "--probability",
        type=float,
        dest="probability",
        default=0.5,
        help="Probability of edge between nodes in training graph",
    )
    parser.add_argument(
        "-C",
        "--colors",
        type=int,
        dest="colors",
        default=24,
        help="Number of colors allowed when training",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="training_logs",
        help="Directory to save training logs and plots"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Detailed evaluation every N epochs (0 to disable)"
    )

    args = parser.parse_args()

    nodes = args.nodes
    probability = args.probability
    colors = args.colors
    tabucol_init = False

    register(
        id="GcpEnvMaxIters-v0",
        entry_point="gcp_env:GcpEnv",
        max_episode_steps=args.max_steps,
    )
    spec = gym.spec("GcpEnvMaxIters-v0")

    env = gym.make(
        spec,
        graph=nx.gnp_random_graph(nodes, probability),
        k=colors,
        tabucol_init=tabucol_init,
    )

    print("Setting up environments...")

    train_envs = SubprocVectorEnv(
        [
            lambda: gym.make(
                spec,
                graph=nx.gnp_random_graph(nodes, probability),
                k=colors,
                tabucol_iters=args.tabucol_iters,
                tabucol_init=tabucol_init,
            )
            for _ in range(10)
        ]
    )
    test_envs = SubprocVectorEnv(
        [
            lambda: gym.make(
                spec,
                graph=nx.gnp_random_graph(nodes, probability),
                k=colors,
                tabucol_iters=args.tabucol_iters,
                tabucol_init=tabucol_init,
            )
            for _ in range(10)
        ]
    )

    print("Setting up policy...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    node_features = 3
    col_features = 3

    print("Setting up networks...")
    actor = ActorNetwork(node_features, col_features, device=device).to(device)
    critic = CriticNetwork(node_features, col_features, device=device).to(device)
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)

    dist = torch.distributions.Categorical
    policy = GCPPPOPolicy(
        actor, critic, optim, dist, k=colors, nodes=nodes, action_space=env.action_space
    )

    if args.input is not None:
        print(f"Loading policy: {args.input}")
        policy.load_state_dict(torch.load(args.input, map_location=device))

    print("Setting up replay buffer and collectors...")

    replay_buffer = VectorReplayBuffer(30000, len(train_envs))
    train_collector = Collector(policy, train_envs, replay_buffer)
    test_collector = Collector(policy, test_envs)

    print("Training...")
    print(f"Configuration: epochs={args.epochs}, save_interval={args.save_interval}")
    print("-" * 70)

    logger = TrainingMonitorLogger(save_dir=args.log_dir)
    
    best_test_score = [float('inf')]
    best_test_reward = [float('-inf')]
    current_epoch = [0]
    last_losses = {}
    
    def save_best_fn(policy):
        best_path = args.output.replace('.pt', '_best.pt')
        torch.save(policy.state_dict(), best_path)
        print(f"  >>> New best model saved to: {best_path}")
    
    def save_checkpoint_fn(epoch, env_step, gradient_step):
        current_epoch[0] = epoch
        
        if epoch % args.save_interval == 0:
            checkpoint_path = args.output.replace('.pt', f'_epoch{epoch}.pt')
            torch.save(policy.state_dict(), checkpoint_path)
            print(f"  >>> Checkpoint saved: {checkpoint_path}")
        
        if args.eval_interval > 0 and epoch % args.eval_interval == 0:
            print(f"\n[Epoch {epoch}] Evaluating on test set...")
            
            # 核心改动：传入 test_collector 而不是 test_envs
            # 让 Collector 来处理所有 obs 转换和 action 映射
            test_result = evaluate_policy_detailed(
                policy, test_collector, test_envs,
                num_episodes=10, verbose=True
            )
            
            if test_result:
                print(f"\n[Epoch {epoch}] Summary:")
                print(f"  Initial Score: {test_result['initial_score_mean']:.1f}")
                print(f"  RL Improvement: {test_result['rl_improvement']:.1f}")
                print(f"  TabuCol Improvement: {test_result['tabucol_improvement']:.1f}")
                print(f"  Final Score: {test_result['test_score_mean']:.1f}")
                print(f"  Improvement Ratio: {test_result['improvement_ratio']:.2%}")
                
                metrics_to_record = {
                    'test_score_mean': test_result['test_score_mean'],
                    'test_score_min': test_result['test_score_min'],
                    'test_reward_mean': test_result['test_reward_mean'],
                    'initial_score_mean': test_result['initial_score_mean'],
                    'rl_improvement': test_result['rl_improvement'],
                    'tabucol_improvement': test_result['tabucol_improvement'],
                    'improvement_ratio': test_result['improvement_ratio'],
                    'perfect_rate': test_result['perfect_rate'],
                }
                
                if last_losses:
                    metrics_to_record['policy_loss'] = last_losses.get('loss/clip', 0)
                    metrics_to_record['value_loss'] = last_losses.get('loss/vf', 0)
                    metrics_to_record['entropy'] = last_losses.get('loss/ent', 0)
                
                logger.record_epoch_metrics(epoch, metrics_to_record)
                
                if test_result['test_score_mean'] < best_test_score[0]:
                    best_test_score[0] = test_result['test_score_mean']
                    best_path = args.output.replace('.pt', '_best.pt')
                    torch.save(policy.state_dict(), best_path)
                    print(f"  >>> New best model saved! Score: {test_result['test_score_mean']:.1f}")
            
            print()
        
        return None

    result = onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epochs,
        step_per_epoch=50000,
        repeat_per_collect=10,
        episode_per_test=10,
        batch_size=512,
        step_per_collect=2000,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        test_in_train=False,
        verbose=True,
        logger=logger,
    )

    print("\nTraining complete!")
    
    logger.save_metrics()
    logger.plot_curves()
    logger.print_summary()
    
    torch.save(policy.state_dict(), args.output)
    print(f"Final model saved to: {args.output}")
    
    best_model_path = args.output.replace('.pt', '_best.pt')
    if os.path.exists(best_model_path):
        print(f"Best model available at: {best_model_path}")