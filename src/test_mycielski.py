import argparse
import numpy as np
import networkx as nx
import torch
import gymnasium as gym
from gymnasium.envs.registration import register
from tianshou.data import Collector
from tianshou.utils.net.common import ActorCritic
from tianshou.env import DummyVectorEnv
from network import ActorNetwork, CriticNetwork, GCPPPOPolicy
from tianshou.policy import BasePolicy
from tianshou.data import Batch
import time


class RandomGCPPolicy(BasePolicy):
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def forward(self, batch, state=None, **kwargs):
        action = np.array([self.action_space.sample()])
        return Batch(act=action)

    def learn(self, batch: Batch, **kwargs):
        return {}


def calculate_score(graph, solution):
    score = 0
    for node in graph.nodes():
        for neighbor in graph.neighbors(node):
            if solution[node] == solution[neighbor]:
                score += 1
    score = score // 2
    return score


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test RL policy on Mycielski graphs"
    )
    parser.add_argument("policy", type=str, help="Path to policy to use")
    parser.add_argument(
        "--mycielski-n",
        type=int,
        dest="mycielski_n",
        default=3,
        help="Mycielski graph parameter n (default: 3, gives chi=4)"
    )
    parser.add_argument(
        "-k",
        "--colors",
        type=int,
        dest="colors",
        default=4,
        help="Number of colors (default: 4, matches chi of Mycielski(3))"
    )
    parser.add_argument(
        "-I",
        "--max-steps",
        type=int,
        dest="max_steps",
        default=300,
        help="Max RL steps per episode",
    )
    parser.add_argument(
        "-T",
        "--max-tabucol-iters",
        type=int,
        dest="max_tabucol_iters",
        default=5000,
        help="Max tabucol iterations in each episode",
    )
    parser.add_argument(
        "-E",
        "--max-episodes",
        type=int,
        dest="episodes",
        default=10,
        help="Max episodes to run",
    )
    parser.add_argument(
        "-B",
        "--beta",
        type=float,
        dest="beta",
        default=0.2,
        help="Beta parameter in RLTCol",
    )
    parser.add_argument(
        "--RL", type=str2bool, nargs="?", const=True, default=True, help="Use RL or not"
    )
    parser.add_argument(
        "--disable-tabucol",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Disable TabuCol to test RL-only performance"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    node_features = 3
    col_features = 3

    args = parser.parse_args()

    print("="*70)
    print("Testing RL Policy on Mycielski Graph")
    print("="*70)

    graph = nx.mycielski_graph(n=args.mycielski_n)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    print(f"\nGraph Information:")
    print(f"  Type: Mycielski({args.mycielski_n})")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {num_edges}")
    print(f"  Chromatic number (χ): {args.mycielski_n + 1}")
    print(f"  Colors (k): {args.colors}")

    if args.colors < args.mycielski_n + 1:
        print(f"\n⚠️  WARNING: k={args.colors} < χ={args.mycielski_n + 1}")
        print(f"  Perfect solution (score=0) is impossible!")
    else:
        print(f"\n✓ k={args.colors} >= χ={args.mycielski_n + 1}")
        print(f"  Perfect solution (score=0) is possible!")

    register(
        id="GcpEnvMaxIters-v0",
        entry_point="gcp_env:GcpEnv",
        max_episode_steps=args.max_steps,
    )

    spec = gym.spec("GcpEnvMaxIters-v0")

    env = gym.make(
        spec,
        graph=graph,
        k=args.colors,
        tabucol_iters=args.max_tabucol_iters,
        beta=args.beta,
        tabucol_init=False,
        aco_init=False,
        disable_tabucol=args.disable_tabucol,
    )

    vector_env = DummyVectorEnv([lambda: env])
    policy = None

    if args.RL:
        actor = ActorNetwork(node_features, col_features, device=device).to(device)
        critic = CriticNetwork(node_features, col_features, device=device).to(device)
        actor_critic = ActorCritic(actor, critic).to(device)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)

        dist = torch.distributions.Categorical
        policy = GCPPPOPolicy(
            actor,
            critic,
            optim,
            dist,
            k=args.colors,
            nodes=num_nodes,
            action_space=env.action_space,
        ).to(device)

        policy.load_state_dict(torch.load(args.policy, map_location=device))
        print(f"\nPolicy loaded: {args.policy}")
    else:
        policy = RandomGCPPolicy(action_space=env.action_space)
        print(f"\nUsing random policy")

    policy.eval()
    eval_collector = Collector(policy, vector_env)
    eval_collector.reset()

    print("\n" + "-"*70)
    print("Starting evaluation")
    print("-"*70)
    print(f"Device: {device}")
    print(f"Max steps: {args.max_steps}")
    print(f"Max episodes: {args.episodes}")
    print(f"TabuCol: {'Disabled' if args.disable_tabucol else 'Enabled'}")
    print(f"Max TabuCol iters: {args.max_tabucol_iters}")
    print("-"*70)

    episodes = 0
    best_score = float('inf')
    best_solution = None
    all_scores = []
    all_rl_improvements = []
    all_tabucol_improvements = []

    while episodes < args.episodes:
        episodes += 1
        start_time = time.time()

        result = eval_collector.collect(n_episode=1)

        end_time = time.time()

        solution = env.env.get_solution()
        current_score = calculate_score(graph, solution)

        initial_score = env.env.last_episode_initial_score
        rl_improvement = env.env.last_episode_rl_improvement
        tabucol_improvement = env.env.last_episode_tabucol_improvement

        all_scores.append(current_score)
        all_rl_improvements.append(rl_improvement)
        all_tabucol_improvements.append(tabucol_improvement)

        if current_score < best_score:
            best_score = current_score
            best_solution = solution.copy()

        print(f"\nEpisode {episodes}:")
        print(f"  Initial Score: {initial_score}")
        print(f"  RL Improvement: {rl_improvement:+.1f}")
        print(f"  TabuCol Improvement: {tabucol_improvement:+.1f}")
        print(f"  Final Score: {current_score}")
        print(f"  Time: {end_time - start_time:.2f}s")

        if current_score == 0:
            print(f"\n🎉 Perfect solution found in episode {episodes}!")
            break

    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Total episodes: {episodes}")
    print(f"\nScore Statistics:")
    print(f"  Best Score: {best_score}")
    print(f"  Average Score: {np.mean(all_scores):.2f}")
    print(f"  Std Dev: {np.std(all_scores):.2f}")
    print(f"  Min Score: {np.min(all_scores)}")
    print(f"  Max Score: {np.max(all_scores)}")

    print(f"\nRL Performance:")
    print(f"  Average RL Improvement: {np.mean(all_rl_improvements):.2f}")
    print(f"  Best RL Improvement: {np.max(all_rl_improvements):.2f}")
    print(f"  Worst RL Improvement: {np.min(all_rl_improvements):.2f}")

    if not args.disable_tabucol:
        print(f"\nTabuCol Performance:")
        print(f"  Average TabuCol Improvement: {np.mean(all_tabucol_improvements):.2f}")
        print(f"  Best TabuCol Improvement: {np.max(all_tabucol_improvements):.2f}")
        print(f"  Worst TabuCol Improvement: {np.min(all_tabucol_improvements):.2f}")

    print(f"\nSuccess Rate:")
    perfect_count = sum(1 for s in all_scores if s == 0)
    print(f"  Perfect Solutions: {perfect_count}/{episodes} ({perfect_count/episodes*100:.1f}%)")

    if best_score == 0:
        print(f"\n✓ Successfully found perfect solution!")
    else:
        print(f"\n⚠️  Best score: {best_score} (not perfect)")

    print("="*70)
