import torch
import numpy as np
from agent.replay_buffer import ReplayBuffer

class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, gamma=0.95, batch_size=64,
                 max_epsilon=0.9, min_epsilon=0.1, eps_decay=100,
                 tau=0.01, lr=1e-2, buffer_size=1e5):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            gamma: discount factor of future rewards.
            batch_size: Number of samples per batch.
            tao: indicates the speed of adjustment of the slowly updated target network.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
            lr: learning rate of the optimizer
        """
        # setup networks
        self.Q = Q.cuda()
        self.Q_target = Q_target.cuda()
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.eps_decay = eps_decay
        self.n_steps = 0

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # TODO:
        # 1. add current transition to replay buffer
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)

        # 2. sample next batch and perform batch update:
        b_states, b_actions, b_next_states, b_rewards, b_dones = self.replay_buffer.next_batch(self.batch_size)
        b_states = torch.tensor(b_states).float().cuda()
        b_actions = torch.tensor(b_actions).long().view(b_actions.shape[0], 1).cuda()
        b_next_states = torch.tensor(b_next_states).float().cuda()
        b_rewards = torch.tensor(b_rewards).float().cuda()
        b_dones = 1.0-torch.tensor(b_dones).float().cuda()

        # 2.1 compute td targets and loss
        #         td_target =  reward + discount * max_a Q_target(next_state_batch, a)

        # getting actions from current state - predictions
        q_predictions = self.Q(b_states).gather(1, b_actions).squeeze(1)
        # generating actions from Q for next state (double-Q)
        double_q_actions = torch.argmax(self.Q(b_next_states), dim=1)
        # getting value from Q_target for actions from Q (double-Q)
        q_target_values = self.Q_target(b_next_states).detach()[np.arange(self.batch_size), double_q_actions]
        # compute td target (accounting for terminal states)
        td_target = b_rewards + self.gamma * q_target_values * b_dones

        # 2.2 update the Q network
        self.update(q_predictions, td_target)

        # 2.3 call soft update for target network
        self.soft_update()

    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)    
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """

        # update epsilon
        eps = self.min_epsilon + (self.max_epsilon - self.min_epsilon)*np.exp(-1*self.n_steps/self.eps_decay)
        self.n_steps += 1
        eps = self.min_epsilon

        r = np.random.uniform()
        if deterministic or r > eps:
            # TODO: take greedy action (argmax)
            state = torch.tensor([state]).float().cuda()
            action_id = torch.argmax(self.Q(state), dim=1).cpu().detach().numpy()
        else:

            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work.
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            action_id = np.random.choice(range(self.num_actions), 1)

        return action_id[0]

    def update(self, y_hat, y):
        """
        Runs 1 update step on Q network
        """
        self.optimizer.zero_grad()
        loss = self.loss_function(y_hat, y)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def soft_update(self):
        """
        Copies parameters from Q network to Q-target network by polyak averaging
        """
        for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))
