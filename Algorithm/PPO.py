import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
import matplotlib
import matplotlib.pyplot as plt

class Actor(nn.Module):
    def __init__(self, n_states,hidden_dim, n_actions):
        """
        Actor network for policy approximation.

        Args:
            n_states (int): Dimension of the state space.
            n_actions (int): Number of hidden units in layers.
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, n_actions)

        self.log_std = nn.Parameter(torch.zeros(1, n_actions))  # Learned log std

        self.init_weights()

    def init_weights(self):
        """
        Initialize network weights using Xavier initialization for better convergence.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()
                # nn.init.xavier_uniform_(layer.weight)  # Xavier initialization
                # nn.init.zeros_(layer.bias)  # Initialize bias to 0

    def forward(self, state):
        """
        Forward pass for action selection.

        Args:
            state (Tensor): Current state of the environment.

        Returns:
            Tensor: Selected action values.
        """
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        mu = self.mu(x)

        std = self.log_std.exp()
        dist = torch.distributions.normal.Normal(mu, std)

        return dist

class Critic(nn.Module):
    def __init__(self, n_states, hidden_dim, learning_rate=1e-4):
        """
        Critic network for state-value approximation.

        Args:
            state_dim (int): Dimension of the state space.
            hidden_dim (int): Number of hidden units in layers.
            learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-4.
        """
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(n_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, out_features=1)

        # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.init_weights()

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, state):
        """
        Forward pass for state-value estimation.

        Args:
            state (Tensor): Current state of the environment.

        Returns:
            Tensor: Estimated V(s) value.
        """
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))

        value = self.value(x)

        return value
    
class PPO:
    def __init__(self, 
                device = None, 
                num_of_action: int = 2,
                action_range: list = [-2.5, 2.5],
                n_observations: int = 6,
                hidden_dim = 64,
                dropout = 0.05, 
                learning_rate: float = 0.01,
                tau: float = 0.005,
                discount_factor: float = 0.95,
                buffer_size: int = 256,
                batch_size: int = 1,
                clip: float = 0.2
                ):
        """
        Actor-Critic algorithm implementation.

        Args:
            device (str): Device to run the model on ('cpu' or 'cuda').
            num_of_action (int, optional): Number of possible actions. Defaults to 2.
            action_range (list, optional): Range of action values. Defaults to [-2.5, 2.5].
            n_observations (int, optional): Number of observations in state. Defaults to 4.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 256.
            learning_rate (float, optional): Learning rate. Defaults to 0.01.
            tau (float, optional): Soft update parameter. Defaults to 0.005.
            discount_factor (float, optional): Discount factor for Q-learning. Defaults to 0.95.
            batch_size (int, optional): Size of training batches. Defaults to 1.
            buffer_size (int, optional): Replay buffer size. Defaults to 256.
        """
        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.device = device
        self.action_range = action_range
        self.num_of_action = num_of_action

        self.actor = Actor(n_observations, hidden_dim, num_of_action).to(device)
        self.critic = Critic(n_observations,  hidden_dim).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=learning_rate, eps=1e-5)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=learning_rate, eps=1e-5)
        self.critic_loss = torch.nn.MSELoss()
        
        # self.scheduler = lambda step: max(1.0 - float(step / self.episode_durations), 0) ### lambda value decay method

        # self.actor_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=self.scheduler)
        # self.critic_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=self.scheduler)
 
        self.batch_size = batch_size
        self.tau = tau
        self.discount_factor = discount_factor
        self.clip = clip
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        # self.update_target_networks(tau=1)  # initialize target networks
        self.episode_durations = []

        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.

        pass
        # ====================================== #

    def select_action(self, state, noise=0.0):
        """
        Selects an action based on the current policy with optional exploration noise.
        
        Args:
        state (Tensor): The current state of the environment.
        noise (float, optional): The standard deviation of noise for exploration. Defaults to 0.0.

        Returns:
            Tuple[Tensor, Tensor]: 
                - scaled_action: The final action after scaling.
                - clipped_action: The action before scaling but after noise adjustment.
        """
        with torch.no_grad():
            dist = self.actor(state)
            # probs = torch.clamp(probs, min=1e-6, max=1.0) # avoid log(0) or NaN sampling
            # dist = torch.distributions.Categorical(probs)
            # print(dist)
            action = dist.sample()
            # print(action)
            # action_clipped = action.clamp(*self.action_range)
            log_prob = dist.log_prob(action).sum(dim=-1)
            # print(action_clipped,log_prob)
        return action, log_prob ,dist.entropy().sum(dim=-1)
        # ====================================== #
    
    def generate_sample(self, batch_size):
        """
        Generates a batch sample from memory for training.

        Returns:
            Tuple: A tuple containing:
                - state_batch (Tensor): The batch of current states.
                - action_batch (Tensor): The batch of actions taken.
                - reward_batch (Tensor): The batch of rewards received.
                - next_state_batch (Tensor): The batch of next states received.
                - done_batch (Tensor): The batch of dones received.
        """
        # Ensure there are enough samples in memory before proceeding
        # Sample a batch from memory
        # ========= put your code here ========= #
        if len(self.memory) < batch_size:
            return None
        batch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, log_prob_batch, value_batch= zip(*batch)
        return (
            torch.stack([torch.tensor(s, dtype=torch.float32) for s in state_batch]).to(self.device),
            torch.tensor(action_batch, dtype=torch.float32, device=self.device),
            torch.tensor(reward_batch, dtype=torch.float32, device=self.device),
            torch.stack([torch.tensor(s, dtype=torch.float32) for s in next_state_batch]).to(self.device),
            torch.tensor(done_batch, dtype=torch.bool, device=self.device),
            torch.tensor(log_prob_batch, dtype=torch.float32, device=self.device),
            torch.tensor(value_batch, dtype=torch.float32, device=self.device),
        )
        # ====================================== #

    def calculate_loss(self, states, actions, rewards, dones, old_log_probs, advantages):
        """
        Computes the loss for policy optimization.

        Args:
            - states (Tensor): The batch of current states.
            - actions (Tensor): The batch of actions taken.
            - rewards (Tensor): The batch of rewards received.
            - next_states (Tensor): The batch of next states received.
            - dones (Tensor): The batch of dones received.

        Returns:
            Tensor: Computed critic & actor loss.
        """
        # Update Critic
        values = self.critic(states)
        critic_loss = self.critic_loss(values, rewards)
        # Gradient clipping for critic

        # Update Actor
        dist = self.actor(states)                        # shape: (batch_size, num_actions)
        new_log_probs = dist.log_prob(dist)             # shape: (batch_size,)
        old_log_probs = old_log_probs.detach() # From traject

        # Gradient clipping for actor
        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
        actor_loss = -torch.min(surrogate1, surrogate2).mean()

        return actor_loss, critic_loss
    
    def get_gae(self,rewards, values, dones, gamma=0.99, lam=0.95):

        advs = []
        gae = 0


        for step in reversed(range(len(rewards))):
            done_mask = 1.0 - dones[step].float()
            delta = rewards[step] + gamma * values[step + 1] * done_mask - values[step]
            gae = delta + gamma * lam * done_mask * gae
            advs.insert(0, gae)

        return torch.tensor(advs, device=self.device)
    
    def choose_mini_batch(self, batch_size, states, actions, returns, advs, log_probs ,values):
        data_size = len(states)
        indices = np.arange(data_size)
        np.random.shuffle(indices)
        print(states)
        for start in range(0, data_size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            # Convert to torch indices if your data is in tensors
            batch_indices = torch.tensor(batch_indices, dtype=torch.long, device=self.device)

            yield states[batch_indices], actions[batch_indices], returns[batch_indices], \
                advs[batch_indices], log_probs[batch_indices], values[batch_indices]



    def update_policy(self,states, actions, rewards,dones,old_log_probs, values):
        """
        Update the policy using the calculated loss.

        Returns:
            float: Loss value after the update.
        """
        # ========= put your code here ========= #
        
        # sample = self.generate_sample(self.batch_size)
        # if sample is None:
        #     return 0,0
        # states, actions, rewards, next_states, dones,old_log_probs,values = sample
        # actions = actions.long()
        # with torch.no_grad():
        #     next_value = self.critic(next_states)
        # print(next_value)
        # print(values)
        # values = torch.cat([values, next_value.unsqueeze(0)], dim=0)
        

       
            # Estimate state values
            # values = self.critic(states)
            
        #     td_target = rewards + self.discount_factor * next_values * (~dones)
        #     advantages = td_target - values
        advantages = self.get_gae(rewards, values, dones)
        

        # Compute critic and actor loss
        for state, actions, rewards, advantages, old_log_probs, old_values in self.choose_mini_batch(self.batch_size,
                                                                                               states, actions, rewards,
                                                                                               advantages, old_log_probs,values):
            rewards = self.safe_standardize(rewards)
            actor_loss, critic_loss = self.calculate_loss(states, actions, rewards, dones, old_log_probs, advantages)
        
            # Backpropagate and update critic network parameters
        
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            # Backpropagate and update actor network parameters
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
        # ====================================== #
        return actor_loss, critic_loss
    
    def safe_standardize(self,tensor, eps=1e-6):
        std = tensor.std()
        return (tensor - tensor.mean()) / (std + eps) if std > 0 else tensor - tensor.mean()


    # def update_target_networks(self, tau=None):
    #     """
    #     Perform soft update of target networks using Polyak averaging.

    #     Args:
    #         tau (float, optional): Update rate. Defaults to self.tau.
    #     """
    #     # ========= put your code here ========= #
    #     if tau is None:
    #         tau = self.tau

    #     for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
    #         target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def scale_action(self, action):
        action_min, action_max = self.action_range
        return torch.tensor(action_min + (action_max - action_min) * (action / (self.num_of_action - 1)), dtype=torch.float32)

    def learn(self, env, max_steps, num_agents, noise_scale=0.1, noise_decay=0.99):
        """
        Train the agent on a single step.

        Args:
            env: The environment in which the agent interacts.
            max_steps (int): Maximum number of steps per episode.
            num_agents (int): Number of agents in the environment.
            noise_scale (float, optional): Initial exploration noise level. Defaults to 0.1.
            noise_decay (float, optional): Factor by which noise decreases per step. Defaults to 0.99.
        """

        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        obs,_ = env.reset()
        total_reward = 0
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        # ====================================== #

        for step in range(max_steps):

            state = obs['policy']
            action, log_prob,_ = self.select_action(state)
            print(action)
            # scaled_action = self.scale_action(action)
            # print(scaled_action)
            action_tensor = torch.tensor([[action]], dtype=torch.float32)
            value = self.critic(state)
            next_obs, reward, terminated, truncated, _ = env.step(action_tensor)
            next_state = next_obs['policy']
            done = terminated or truncated

            # self.memory.append((state, action, reward, next_state, done, log_prob,value))
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(done)
            # Parallel Agents Training
            if num_agents > 1:
                pass
            # Single Agent Training
            else:
                pass
            # ====================================== #

            # Update state
            total_reward += reward
            obs = next_obs
            # Decay the noise to gradually shift from exploration to exploitation
            if done:
                self.plot_durations(step)
                break

            # Perform one step of the optimization (on the policy network)
        next_value = self.critic(next_state)
        values.append(next_value)
        actor_loss, critic_loss = self.update_policy(states, actions, rewards, dones,log_probs,values)

            # Update target networks
        # self.update_target_networks()

        return total_reward,step,actor_loss, critic_loss

    def plot_durations(self, timestep=None, show_result=False):
        if timestep is not None:
            self.episode_durations.append(timestep)

        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        # if self.is_ipython:
        #     if not show_result:
        #         display.display(plt.gcf())
        #         display.clear_output(wait=True)
        #     else:
        #         display.display(plt.gcf())
    # ================================================================================== #