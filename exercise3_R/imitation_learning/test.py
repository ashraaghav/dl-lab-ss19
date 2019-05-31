# from __future__ import print_function

import sys
sys.path.append("../") 

from datetime import datetime
import numpy as np
import gym
import os
import json
import torch
import argparse

from agent.bc_agent import BCAgent
from utils import *


def run_episode(env, agent, history_length=1, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    init_accelerate = 5

    state = env.reset()
    
    # fix bug of curropted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events() 

    # preprocessing parameters
    state_history = []

    while True:

        # TODO: preprocess the state in the same way than in your preprocessing in train_agent.py
        # converting to grayscale
        state = rgb2gray(state)

        # Removing 'score' information - it is probably noise??
        # state[85:, :15] = 0.0

        # recording history
        state_history.append(state)

        if history_length < len(state_history):
            state_history = state_history[-history_length:]
            state = torch.tensor([state_history]).float().to(agent.device)
        else:
            # initial frames - set accelerating forward by default ??
            state = None

        # TODO: get the action from your agent! You need to transform the discretized actions to continuous
        # actions.
        # hints:
        #       - the action array fed into env.step() needs to have a shape like np.array([0.0, 0.0, 0.0])
        #       - just in case your agent misses the first turn because it is too fast: you are allowed to clip the acceleration in test_agent.py
        #       - you can use the softmax output to calculate the amount of lateral acceleration

        if state is not None:
            pred = agent.predict(state)
        else:
            pred = 3

        a = id_to_action(pred)

        print(a)

        # if car stopped for 'n' consecutive frames, restart by accelerating
        if len(state_history) >= history_length and np.all(state_history[-1][:85, :] == state_history[-3][:85, :]):
            print('manual acceleration....')
            a[1] = 0.8

        # accelerating initially regardless of input since this was not reflected in model
        # if init_accelerate > 0:
        #     a[1] = 1.0
        #     init_accelerate -= 1

        next_state, r, done, info = env.step(a)
        episode_reward += r
        state = next_state
        step += 1

        if rendering:
            env.render()

        if done or step > max_timesteps:
            break

    return episode_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model file to use", required=True)
    parser.add_argument("--history", type=int, help="history length in model", required=True)

    args = parser.parse_args()

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 5           # number of episodes to test
    history_length = args.history

    # TODO: load agent
    agent = BCAgent(device='cuda', history_length=history_length, n_classes=5)
    agent.load(args.model)

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, history_length=history_length,
                                     rendering=rendering)
        print('Episode reward: ', episode_reward)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
