from __future__ import print_function

import gym
import numpy as np
import os
import argparse
from datetime import datetime
import json

from agent.dqn_agent import DQNAgent
from train_carracing import run_episode
from agent.networks import *

np.random.seed(0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Model file to use", required=True)
    parser.add_argument('-e', "--episodes", type=int, help="num episodes to try", default=5, required=False)
    args = parser.parse_args()

    env = gym.make("CarRacing-v0").unwrapped

    history_length =  5

    #TODO: Define networks and load agent
    # ....
    Q_network = CNN(history_length=history_length, n_classes=5)
    Q_target = CNN(history_length=history_length, n_classes=5)
    agent = DQNAgent(Q=Q_network, Q_target=Q_target, num_actions=5)
    agent.load(args.model)

    episode_rewards = []
    for i in range(args.episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True, history_length=history_length)
        episode_rewards.append(stats.episode_reward)
        print('Episode %d - [ Reward %.2f ]' % (i+1, stats.episode_reward))

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    if not os.path.exists("./results"):
        os.mkdir("./results")  

    fname = "./results/carracing_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
