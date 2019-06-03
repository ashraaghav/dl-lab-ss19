# from __future__ import print_function

import sys
sys.path.append("../")

import argparse
from pyglet.window import key
import gym
import numpy as np
import pickle
import os
from datetime import datetime
import gzip
import json

import utils


def key_press(k, mod):
    global restart
    if k == key.Q: restart = True
    # if k == key.J:  a[0] = -1.0
    # if k == key.L: a[0] = +1.0
    # if k == key.I:    a[1] = +1.0
    # if k == key.K:  a[2] = +0.2
    if k == key.LEFT:  a[0] = -1.0
    if k == key.RIGHT: a[0] = +1.0
    if k == key.UP:    a[1] = +1.0
    if k == key.DOWN:  a[2] = +0.2

def key_release(k, mod):
    # if k == key.J and a[0] == -1.0: a[0] = 0.0
    # if k == key.L and a[0] == +1.0: a[0] = 0.0
    # if k == key.I:    a[1] = 0.0
    # if k == key.K:  a[2] = 0.0
    if k == key.LEFT and a[0] == -1.0: a[0] = 0.0
    if k == key.RIGHT and a[0] == +1.0: a[0] = 0.0
    if k == key.UP:    a[1] = 0.0
    if k == key.DOWN:  a[2] = 0.0


def store_data(data, datasets_dir="./data"):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
    f = gzip.open(data_file,'wb')
    pickle.dump(data, f)


def save_results(episode_rewards, results_dir="./results"):
    # save results
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

     # save statistics in a dictionary and write them into a .json file
    results = dict()
    results["number_episodes"] = len(episode_rewards)
    results["episode_rewards"] = episode_rewards

    results["mean_all_episodes"] = np.array(episode_rewards).mean()
    results["std_all_episodes"] = np.array(episode_rewards).std()
 
    fname = os.path.join(results_dir, "results_manually-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S"))
    fh = open(fname, "w")
    json.dump(results, fh)
    print('... finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect_data", action="store_true", default=False, help="Collect the data in a pickle file.")

    args = parser.parse_args()

    samples = {
        "state": [],
        "next_state": [],
        "reward": [],
        "action": [],
        "terminal" : [],
    }

    env = gym.make('CarRacing-v0').unwrapped

    env.reset()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    sample_updated = False

    a = np.array([0.0, 0.0, 0.0]).astype('float32')
    
    episode_rewards = []
    steps = 0
    while True:
        episode_reward = 0
        epi_samples = {
            "state": [],
            "next_state": [],
            "reward": [],
            "action": [],
            "terminal" : [],
        }
        state = env.reset()
        while True:

            next_state, r, done, info = env.step(a)
            episode_reward += r

            # converting to grayscale to save image quicker...
            state = utils.rgb2gray(state)

            epi_samples["state"].append(state)            # state has shape (96, 96)
            epi_samples["action"].append(np.array(a))     # action has shape (1, 3)
            epi_samples["next_state"].append(next_state)
            epi_samples["reward"].append(r)
            epi_samples["terminal"].append(done)

            state = next_state
            steps += 1

            if steps == 1000 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("\nstep {}".format(steps))
                steps = 0
                break

            # if args.collect_data and steps % 5000 == 0:
            #     print('... saving data')
            #     store_data(samples, "./data")
            #     save_results(episode_rewards, "./results")

            env.render()
            if done: 
                break

        print("Episode reward: %.2f" % episode_reward)
        # update samples only if reward is high
        if episode_reward > 850:
            episode_rewards.append(episode_reward)
            samples["state"].extend(epi_samples["state"])
            samples["action"].extend(epi_samples["action"])
            samples["next_state"].extend(epi_samples["next_state"])
            samples["reward"].extend(epi_samples["reward"])
            samples["terminal"].extend(epi_samples["terminal"])

            sample_updated = True

        if args.collect_data and sample_updated and len(episode_rewards) % 5 == 0:
            print('... saving data. size: ', len(samples['state']))
            store_data(samples, "./data")
            save_results(episode_rewards, "./results")
            sample_updated = False

    env.close()

    

   
