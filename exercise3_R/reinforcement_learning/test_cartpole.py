import os
from datetime import datetime
import gym
import json
import argparse

from agent.dqn_agent import DQNAgent
from train_cartpole import run_episode
from agent.networks import *
import numpy as np

np.random.seed(0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Model file to use", required=True)
    parser.add_argument('-e', "--episodes", type=int, help="num episodes to try", default=5, required=False)
    args = parser.parse_args()

    env = gym.make("CartPole-v0").unwrapped

    # TODO: load DQN agent

    state_dim = 4
    num_actions = 2

    Q_network = MLP(state_dim, num_actions)
    Q_target_network = MLP(state_dim, num_actions)
    agent = DQNAgent(Q=Q_network, Q_target=Q_target_network, num_actions=num_actions)
    agent.load(args.model)
 
    n_test_episodes = args.episodes

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True, max_timesteps=250)
        episode_rewards.append(stats.episode_reward)
        print('Episode %d (reward: %d)' % (i, stats.episode_reward))

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    if not os.path.exists("./results"):
        os.mkdir("./results")  

    fname = "./results/cartpole_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print('... finished')

