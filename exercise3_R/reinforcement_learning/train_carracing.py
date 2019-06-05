# export DISPLAY=:0 

import sys

sys.path.append("../")

import numpy as np
import gym
from agent.dqn_agent import DQNAgent
from agent.networks import CNN
from tensorboard_evaluation import *
import itertools as it
from datetime import datetime
from utils import *
import argparse


def run_episode(env, agent, deterministic, skip_frames=0, do_training=True, rendering=False,
                max_timesteps=1000, history_length=0):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events()

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * history_length)
    state = np.array(image_hist)#.reshape(96, 96, history_length)

    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly. 
        # action_id = agent.act(...)
        # action = your_id_to_action_method(...)
        action_id = agent.act(state, deterministic)
        action = id_to_action(action_id)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal:
                break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist)#.reshape(96, 96, history_length)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps:
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, history_length=0, eval_cycle=20, num_eval_episodes=20,
                 skip_frames=0, max_timesteps=1000, min_timesteps=100, timestep_decay=100,
                 model_dir="./models_carracing", tensorboard_dir="./tensorboard",
                 rendering=False, save_interrupt=False):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")
    idx = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), name='RL - Carracing', idx=idx,
                             stats=["episode_reward", "straight", "left", "right", "accel", "brake"])
    tensorboard_eval = Evaluation(os.path.join(tensorboard_dir, "eval"), name='RL - Carracing', idx=idx,
                                  stats=["episode_reward", "straight", "left", "right", "accel", "brake"])

    try:
        for i in range(num_episodes):

            # Hint: you can keep the episodes short in the beginning by changing max_timesteps
            # (otherwise the car will spend most of the time out of the track)
            timesteps = int((min_timesteps) * np.exp((i+1) / timestep_decay))
            timesteps = min(max_timesteps, timesteps)
            # print(timesteps)
            stats = run_episode(env, agent, max_timesteps=timesteps, history_length=history_length,
                                skip_frames=skip_frames, deterministic=False, do_training=True, rendering=rendering)

            print("epsiode %d: %d steps - [ reward %.2f ]" % (i, timesteps, stats.episode_reward))

            tensorboard.write_episode_data(i, eval_dict={"episode_reward": stats.episode_reward,
                                                         "straight": stats.get_action_usage(STRAIGHT),
                                                         "left": stats.get_action_usage(LEFT),
                                                         "right": stats.get_action_usage(RIGHT),
                                                         "accel": stats.get_action_usage(ACCELERATE),
                                                         "brake": stats.get_action_usage(BRAKE)
                                                         })

            # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to
            # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
            # ...
            # if i % eval_cycle == 0:
            #    for j in range(num_eval_episodes):
            #       ...
            if (i + 1) % eval_cycle == 0:
                print("==="*40)
                print("..... EVALUATING .....")
                eval_rewards = []
                for j in range(num_eval_episodes):
                    eval_stats = run_episode(env, agent, deterministic=True, history_length=history_length,
                                             do_training=False, rendering=rendering)
                    eval_rewards.append(eval_stats.episode_reward)
                    print("Eval episode %d: steps %d - [ reward: %d ]" % (j, len(eval_stats.actions_ids),
                                                                        eval_stats.episode_reward))
                tensorboard_eval.write_episode_data(i, eval_dict={"episode_reward": np.mean(eval_rewards),
                                                                  "straight": None,
                                                                  "left": None,
                                                                  "right": None,
                                                                  "accel": None,
                                                                  "brake": None
                                                                  })
                print("---" * 40)

            # store model.
            if (i+1) % eval_cycle == 0 or (i+1) >= num_episodes:
                agent.save(os.path.join(model_dir, "dqn_agent_" + idx + ".pt"))

    except KeyboardInterrupt:
        if save_interrupt:
            agent.save(os.path.join(model_dir, "dqn_agent_" + idx + ".pt"))

    tensorboard.close_session()


def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interrupt", action='store_true', help="Save model if interrupted",
                        default=False, required=False)
    parser.add_argument('-e', "--episodes", type=int, help="num episodes to try", default=500, required=False)
    parser.add_argument("-r", "--render", action='store_true', help="render during training and evaluation",
                        default=False, required=False)
    args = parser.parse_args()
    print(args)

    env = gym.make('CarRacing-v0').unwrapped

    # TODO: Define Q network, target network and DQN agent
    # ...
    Q_network = CNN(history_length=5, n_classes=5)
    Q_target = CNN(history_length=5, n_classes=5)
    agent = DQNAgent(Q=Q_network, Q_target=Q_target, num_actions=5, buffer_size=1e5, lr=1e-4)

    train_online(env, agent, num_episodes=args.episodes, history_length=5, model_dir="./models_carracing",
                 eval_cycle=20, num_eval_episodes=5, skip_frames=5, rendering=args.render,
                 save_interrupt=args.interrupt)
