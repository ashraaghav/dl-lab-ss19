import sys

sys.path.append("../")

import numpy as np
import gym
import itertools as it
import argparse
from datetime import datetime
from agent.dqn_agent import DQNAgent
from tensorboard_evaluation import *
from agent.networks import MLP
from utils import EpisodeStats


def run_episode(env, agent, deterministic, do_training=True, rendering=False, max_timesteps=1000):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()  # save statistics like episode reward or action usage
    state = env.reset()

    step = 0
    while True:

        action_id = agent.act(state=state, deterministic=deterministic)
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        state = next_state

        # # NOTE reward shaping...
        # if terminal:
        #     reward += -1
        #     if step < 20:
        #         reward += -10
        # if step > 100:
        #     reward += 10

        stats.step(reward, action_id)

        if rendering:
            env.render()

        if terminal or step > max_timesteps:
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, max_timesteps, eval_cycle, num_eval_episodes, rendering=False,
                 model_dir="./models_cartpole", tensorboard_dir="./tensorboard", save_interrupt=False):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")

    idx = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), name='RL - Cartpole', idx=idx,
                             stats=["episode_reward", "a_0", "a_1"])
    tensorboard_eval = Evaluation(os.path.join(tensorboard_dir, "eval"), name='RL - Cartpole', idx=idx,
                                  stats=["episode_reward", "a_0", "a_1"])

    try:
        # training
        for i in range(num_episodes):
            stats = run_episode(env, agent, deterministic=False, do_training=True, rendering=rendering,
                                max_timesteps=max_timesteps)
            print("episode: %d (reward: %d) Action usage: %.4f %.4f" % (i, stats.episode_reward,
                                                                         stats.get_action_usage(0),
                                                                         stats.get_action_usage(1)))
            tensorboard.write_episode_data(i, eval_dict={"episode_reward": stats.episode_reward,
                                                         "a_0": stats.get_action_usage(0),
                                                         "a_1": stats.get_action_usage(1)})

            # TODO: evaluate your agent every 'eval_cycle' episodes
            # using run_episode(env, agent, deterministic=True, do_training=False) to check its performance
            # with greedy actions only. You can also use tensorboard to plot the mean episode reward.
            if (i+1) % eval_cycle == 0:
                print("... evaluating ...")
                eval_rewards = []
                for j in range(num_eval_episodes):
                    eval_stats = run_episode(env, agent, deterministic=True, do_training=False,
                                             rendering=rendering, max_timesteps=max_timesteps)
                    eval_rewards.append(eval_stats.episode_reward)
                    print("Eval episode: %d (reward: %d) Action usage: %.4f %.4f" % (j, eval_stats.episode_reward,
                                                                                     eval_stats.get_action_usage(0),
                                                                                     eval_stats.get_action_usage(1)))
                tensorboard_eval.write_episode_data(i, eval_dict={"episode_reward": np.mean(eval_rewards),
                                                                  "a_0": np.nan,
                                                                  "a_1": np.nan})

            # store model.
            if i % eval_cycle == 0 or i >= (num_episodes - 1):
                agent.save(os.path.join(model_dir, "dqn_agent_"+idx+".pt"))

    except KeyboardInterrupt:
        if save_interrupt:
            # save model if keyboard iterrupt
            print('saving model since interrupted...')
            agent.save(os.path.join(model_dir, "dqn_agent_"+idx+".pt"))

    tensorboard.close_session()


if __name__ == "__main__":
    # You find information about cartpole in
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interrupt", action='store_true', help="Save model if interrupted",
                        default=False, required=False)
    parser.add_argument('-e', "--episodes", type=int, help="num episodes to try", default=500, required=False)
    parser.add_argument('-s', "--steps", type=int, help="num steps per episode", default=200, required=False)
    parser.add_argument("-r", "--render", action='store_true', help="render during training and evaluation",
                        default=False, required=False)
    args = parser.parse_args()
    print(args)

    env = gym.make("CartPole-v0").unwrapped

    state_dim = 4
    num_actions = 2

    # TODO: 
    # 1. init Q network and target network (see dqn/networks.py)
    Q_network = MLP(state_dim, num_actions)
    Q_target = MLP(state_dim, num_actions)
    # 2. init DQNAgent (see dqn/dqn_agent.py)
    agent = DQNAgent(Q=Q_network, Q_target=Q_target, num_actions=num_actions, buffer_size=1e5)
    # 3. train DQN agent with train_online(...)
    train_online(env=env, agent=agent, num_episodes=args.episodes, max_timesteps=args.steps,
                 eval_cycle=20, num_eval_episodes=5, rendering=args.render,
                 tensorboard_dir='./tensorboard', save_interrupt=args.interrupt)
