import torch.nn as nn
import torch
import torch.nn.functional as F
from copy import deepcopy
from collections import namedtuple, deque
import numpy as np
import math
from .. import constants as c
from .threshold import Threshold

HP_NORM = 1
SUN_NORM = 200
n_step = 3  # parameters in multiple steps
Network_Update_Frequency=8
Noisy_Update = 10000
Batch_Size = 32
Memory_Size = 100000
Std_Init = 0.3
Gamma = 0.99
Learning_Rate = 1e-3
import gc

def sum_onehot(grid):
    return torch.cat([torch.sum(grid==(i+1), axis=-1).unsqueeze(-1) for i in range(4)], axis=-1)

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init = Std_Init):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Initialization
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.zeros(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.zeros(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_sigma = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('bias_epsilon', torch.zeros(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reduce_noise(self, factor=0.99, min_std=0.01):
        self.weight_sigma.data = torch.clamp(self.weight_sigma.data * factor, min=min_std)
        self.bias_sigma.data = torch.clamp(self.bias_sigma.data * factor, min=min_std)

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            self.reduce_noise() 
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)
    

class QNetwork_rainbow(nn.Module):
    def __init__(self, numLanes, numCols, numPlants, epsilon=0.05, learning_rate=Learning_Rate, device='cpu', use_zombienet=True, use_gridnet=True):
        super(QNetwork_rainbow, self).__init__()
        self.device = device
        self.actionSpace = np.arange(numLanes * numCols * numPlants + 2)
        self.numLanes = numLanes
        self.numCols = numCols
        self.numPlants = numPlants

        self.n_inputs = numLanes * (numCols + 2) * 2 + 1 + numPlants
        self.n_outputs = len(self.actionSpace)
        self.actions = np.arange(numLanes * numCols * numPlants + 2)
        self.learning_rate = learning_rate
        self._grid_size = numLanes * numCols

        # TODO
        self.use_zombienet = use_zombienet
        if use_zombienet:
            self.zombienet_output_size = 1
            self.zombienet = ZombieNet(output_size=self.zombienet_output_size, numLanes=numLanes)
            self.n_inputs += (self.zombienet_output_size - 1) * numLanes

        self.use_gridnet = use_gridnet
        if use_gridnet:
            self.gridnet_output_size=4
            self.gridnet = nn.Linear(self._grid_size, self.gridnet_output_size)
            self.n_inputs += self.gridnet_output_size - self._grid_size
  
        self.feature = nn.Sequential(
            NoisyLinear(self.n_inputs, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            NoisyLinear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        
        # Advantage
        self.advantage_hidden = nn.Sequential(
            NoisyLinear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.advantage = NoisyLinear(64, self.n_outputs)
        
        # Value
        self.value_hidden = nn.Sequential(
            NoisyLinear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.value = NoisyLinear(64, 1)
        
        if self.device == 'cuda':
            self.to(self.device)
            
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                          lr=self.learning_rate)
        
    def decide_action(self, state, mask, epsilon):
        # mask = self.env.mask_available_actions()
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions[mask])
        else:
            action = self.get_greedy_action(state, mask)
        return action
    
    def dist(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        elif isinstance(state, torch.Tensor):
            state = state.to(self.device)
        else:
            raise TypeError("State must be a numpy array or torch tensor")
            
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        batch_size = state.size(0)
        x = self.feature(state)
        
        advantage = self.advantage_hidden(x)
        advantage = self.advantage(advantage)
        advantage = advantage.view(batch_size, self.n_outputs, self.atom_size)
        
        value = self.value_hidden(x)
        value = self.value(value)
        value = value.view(batch_size, 1, self.atom_size)
        
        # Numerical stability treatment
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_atoms = q_atoms - q_atoms.max(dim=-1, keepdim=True)[0]
        
        log_probs = F.log_softmax(q_atoms, dim=-1)
        dist = log_probs.exp()
        return dist

    
    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
                
    def get_greedy_action(self, state, mask):
        qvals = self.get_qvals(state)
        qvals[np.logical_not(mask)] = qvals.min()
        return torch.max(qvals, dim=-1)[1].item()

    def get_qvals(self, state, use_zombienet=True):
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
            state_t = torch.FloatTensor(state).to(device=self.device)
            zombie_grid = state_t[:, self._grid_size:(2 * self._grid_size)].reshape(-1, self.numCols)
            plant_grid = state_t[:, :self._grid_size]
            if self.use_zombienet:
                zombie_grid = self.zombienet(zombie_grid).view(-1, self.zombienet_output_size * self.numLanes)
            else:
                zombie_grid = torch.sum(zombie_grid, axis=1).view(-1, self.numLanes)
            if self.use_gridnet:
                plant_grid = self.gridnet(plant_grid)
            state_t = torch.cat([plant_grid, zombie_grid, state_t[:,2 * self._grid_size:]], axis=1)
        else:
            state_t = torch.FloatTensor(state).to(device=self.device)
            zombie_grid = state_t[self._grid_size:(2 * self._grid_size)].reshape(-1, self.numCols)
            plant_grid = state_t[:self._grid_size]
            if self.use_zombienet:
                zombie_grid = self.zombienet(zombie_grid).view(-1)
            else:
                zombie_grid = torch.sum(zombie_grid, axis=1)
            if self.use_gridnet:
                plant_grid = self.gridnet(plant_grid)
            state_t = torch.cat([plant_grid, zombie_grid, state_t[2 * self._grid_size:]])

        # Feature extraction
        features = self.feature(state_t)
        # Value stream
        value = self.value_hidden(features)
        value = self.value(value)  
        # Advantage stream
        advantage = self.advantage_hidden(features)
        advantage = self.advantage(advantage)
        if len(advantage.shape) == 1:
        # single
            q_values = value.squeeze() + (advantage - advantage.mean())
        else:
        # batch
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class ZombieNet(nn.Module):
    def __init__(self, output_size=1, hidden_size=5, numLanes=c.GRID_Y_LEN):
        super(ZombieNet, self).__init__()
        self.fc1 = nn.Linear(numLanes, output_size)
        # self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc1(x)

class RainbowAgent:
    
    def __init__(self, network, buffer, n_iter = 100000, batch_size=Batch_Size, numLanes =  c.GRID_Y_LEN, numCols = c.GRID_X_LEN, numPlants = 4):
        
        self._grid_size = numLanes * numCols
        self.network = network
        self.target_network = deepcopy(network)
        self.buffer = buffer
        self.threshold = Threshold(seq_length = 100000, start_epsilon=1.0, interpolation="exponential",
                           end_epsilon=0.05)
        self.device ='cuda' if torch.cuda.is_available() else 'cpu'
        self.epsilon = 0
        self.batch_size = batch_size
        self.window = 100
        self.reward_threshold = 30000
        self.initialize()
        # self.player = PlayerQ(env = env, render=False)
        self.numLanes = numLanes
        self.numCols = numCols
        self.actionSpace = np.arange(numLanes * numCols * numPlants + 2)
        
        

    def take_step(self, mask, mode='train'):
        # mask = np.full(self.env.mask_available_actions().size(), True)
        if mode == 'explore':
            if np.random.random()<0.5:
                action=0 # Do nothing
            else:
                action = np.random.choice(self.actionSpace, p = mask / mask.sum())
        else:
            action = self.network.decide_action(self.s_0, mask, epsilon=self.epsilon)
            self.step_count += 1
        
        return action

    def train(self, gamma=Gamma, max_episodes=100000,
              network_update_frequency=32,
              network_sync_frequency=2000,
              evaluate_frequency=500,
              evaluate_n_iter=1000):
        mean_rewards_list = []
        self.gamma = gamma
        # Populate replay buffer
        while self.buffer.burn_in_capacity() < 1:
            done = self.take_step(mode='explore')
            # if done:
            #     self.add_play_to_buffer()
        ep = 0
        training = True
        self.s_0 = self._transform_observation(self.env.reset())


        while training:
            self.rewards = 0
            done = False
            while done == False:
                self.epsilon = self.threshold.epsilon(ep)
                done = self.take_step(mode='train')
                # Update network
                if self.step_count % network_update_frequency == 0:
                    self.update()
                # Sync networks
                if self.step_count % network_sync_frequency == 0:
                    self.target_network.load_state_dict(
                        self.network.state_dict())
                    self.sync_eps.append(ep)

                if self.step_count % Noisy_Update ==0:
                    self.network.reset_noise()
                    self.target_network.reset_noise()
                    
                if done:
                    ep += 1
                    self.training_rewards.append(self.rewards)
                    self.training_loss.append(np.mean(self.update_loss))
                    self.update_loss = []
                    mean_rewards = np.mean(
                        self.training_rewards[-self.window:])
                    self.mean_training_rewards.append(mean_rewards)

                    mean_iteration = np.mean(
                        self.training_iterations[-self.window:])
                    self.mean_training_iterations.append(mean_iteration)
                    mean_rewards_list.append(mean_rewards)
                    print("\rEpisode {:d} Mean Rewards {:.2f}\t\t Mean Iterations {:.2f}\t\t".format(
                        ep, mean_rewards,mean_iteration), end="")
                    if ep >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(
                            ep))
                        break
                    # if (ep%evaluate_frequency) == evaluate_frequency - 1:
                    #     avg_score, avg_iter = evaluate(self.player, self.network, n_iter = evaluate_n_iter, verbose=False)
                    #     self.real_iterations.append(avg_iter)
                    #     self.real_rewards.append(avg_score)

            if ep % 500 == 0:  # clean
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                print(ep)


        np.save('mean_rewards.npy', np.array(mean_rewards_list))

    def calculate_loss(self, batch):
        states, actions, rewards, dones, next_states = [i for i in batch]
        
        rewards_t = torch.FloatTensor(rewards).to(device=self.network.device).reshape(-1,1)
        actions_t = torch.LongTensor(np.array(actions)).reshape(-1,1).to(device=self.network.device)
        dones_t = torch.BoolTensor(dones).to(device=self.network.device)
    
        # DDQN Update
        next_masks = np.array([self._get_mask(s) for s in next_states])
        qvals = torch.gather(self.network.get_qvals(states), 1, actions_t)
        
        qvals_next_pred = self.network.get_qvals(next_states)
        qvals_next_pred[np.logical_not(next_masks)] = qvals_next_pred.min()
        next_actions = torch.max(qvals_next_pred, dim=-1)[1]
        next_actions_t = torch.LongTensor(next_actions).reshape(-1,1).to(device=self.network.device)
        target_qvals = self.target_network.get_qvals(next_states)
        qvals_next = torch.gather(target_qvals, 1, next_actions_t).detach()
        
        qvals_next[dones_t] = 0  # Zero-out terminal states
        # gamma for n step
        expected_qvals = rewards_t + (self.gamma ** self.buffer.n_steps) * qvals_next
        loss = nn.MSELoss()(qvals, expected_qvals)
        del qvals, qvals_next_pred, target_qvals
        return loss
    
    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch)
        loss.backward()
        self.network.optimizer.step()
        try:
            states, actions, rewards, dones, next_states, indices, weights = batch
        except ValueError:
            return
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        current_q = self.network(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # caculate TD error
        td_errors = target_q - current_q
        loss = (td_errors.pow(2) * weights).mean()
        loss.backward()
        self.network.optimizer.step()
        if self.network.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())
        td_errors_numpy = td_errors.detach().cpu().numpy().squeeze()
        self.buffer.update_priorities(td_errors_numpy)
        if self.steps_done % 100 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        #del states, actions, rewards, dones, next_states, weights
        #del current_q, next_q, target_q, td_errors, loss

       
        
    def _transform_observation(self, observation):
        observation = observation.astype(np.float64)
        return np.concatenate([
        observation[:self._grid_size],
        observation[self._grid_size:(2*self._grid_size)]/HP_NORM,
        [observation[2 * self._grid_size]/SUN_NORM],
        observation[2 * self._grid_size+1:]
    ])



    def _get_mask(self, observation):
        empty_cells = np.nonzero((observation[:self._grid_size]==0).reshape(self.numLanes, self.numCols))
        mask = np.zeros(self.env.action_space.n, dtype=bool)
        mask[0] = True
        empty_cells = (empty_cells[0] + self.numLanes * empty_cells[1]) * len(self.env.plant_deck)

        available_plants = observation[-len(self.env.plant_deck):]
        for i in range(len(available_plants)):
            if available_plants[i]:
                idx = empty_cells + i + 1
                mask[idx] = True
        return mask

    def _grid_to_lane(self, grid):
        grid = np.reshape(grid, (self.numLanes, self.numCols))
        return np.sum(grid, axis=1)/HP_NORM
        
    def _save_training_data(self, nn_name):
        np.save(nn_name+"_rewards", self.training_rewards)
        np.save(nn_name+"_iterations", self.training_iterations)
        np.save(nn_name+"_real_rewards", self.real_rewards)
        np.save(nn_name+"_real_iterations", self.real_iterations)
        torch.save(self.training_loss, nn_name+"_loss")
        
    def initialize(self):
        self.training_rewards = []
        self.training_loss = []
        self.training_iterations = []
        self.real_rewards = []
        self.real_iterations = []
        self.update_loss = []
        self.mean_training_rewards = []
        self.mean_training_iterations = []
        self.sync_eps = []
        self.rewards = 0
        self.step_count = 0
        self.s_0 = self._transform_observation(self.env.reset())


class experienceReplayBuffer_rainbow:
    def __init__(self, memory_size=50000, burn_in=10000, alpha=0.6):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.alpha = alpha
        self.n_steps = n_step  
        self.gamma = Gamma
        
        self.n_step_buffer = deque(maxlen=self.n_steps)
        
        self.Buffer = namedtuple('Buffer', 
            field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = deque(maxlen=memory_size)
        self.priorities = deque(maxlen=memory_size)
        self.max_priority = 1.0
        self.last_sampled_indices = None

    def _get_n_step_info(self):
        """Calculate n reward"""
        state = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]
        
        reward = 0
        for i in range(len(self.n_step_buffer)):
            reward += (self.gamma ** i) * self.n_step_buffer[i][2]
        
        # check stop
        done = False
        next_state = None
        for i in range(len(self.n_step_buffer)):
            if self.n_step_buffer[i][3]:
                next_state = self.n_step_buffer[i][4]
                done = True
                break
        
        if not done:
            next_state = self.n_step_buffer[-1][4]
            
        return state, action, reward, done, next_state

    def append(self, state, action, reward, done, next_state):
        # Add the current transition to the n-step buffer
        self.n_step_buffer.append((state, action, reward, done, next_state))
        
        if len(self.n_step_buffer) >= self.n_steps or done:
            state, action, reward, done, next_state = self._get_n_step_info()
            
            # Add to the buffer
            self.replay_memory.append(
                self.Buffer(state, action, reward, done, next_state))
            self.priorities.append(self.max_priority)
        
        if done:
            self.n_step_buffer.clear()

    def sample_batch(self, batch_size=Batch_Size):
        if len(self.replay_memory) < batch_size:
            raise ValueError("缓冲区中的经验不足以采样一个批次")

        # Calculate sampling probability
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Priority-based probability sampling
        indices = np.random.choice(len(self.replay_memory), batch_size, p=probabilities, replace=False)
        samples = [self.replay_memory[i] for i in indices]
        batch = zip(*samples)
        self.last_sampled_indices = indices
        del samples, priorities, probabilities
        return batch

    def update_priorities(self, td_errors):
        if self.last_sampled_indices is None:
            raise ValueError("sample before update_priorities")

        for idx, td_error in zip(self.last_sampled_indices, td_errors):
            priority = abs(td_error) + 1e-6  
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in


# class PlayerQ():
#     def __init__(self, env = None, render=True):
#         if env==None:
#             self.env = gym.make('gym_pvz:pvz-env-v2')
#         else:
#             self.env = env
#         self.render = render
#         self._grid_size = config.N_LANES * config.LANE_LENGTH

#     def get_actions(self):
#         return list(range(self.env.action_space.n))

#     def num_observations(self):
#         return config.N_LANES * config.LANE_LENGTH + config.N_LANES + len(self.env.plant_deck) + 1

#     def num_actions(self):
#         return self.env.action_space.n

#     def _transform_observation(self, observation):
#         observation = observation.astype(np.float64)
#         observation = np.concatenate([observation[:self._grid_size],
#         observation[self._grid_size:(2*self._grid_size)]/HP_NORM,
#         [observation[2 * self._grid_size]/SUN_NORM], 
#         observation[2 * self._grid_size+1:]])
#         return observation

#     def _grid_to_lane(self, grid):
#         grid = np.reshape(grid, (config.N_LANES, config.LANE_LENGTH))
#         return np.sum(grid, axis=1)/HP_NORM

#     def play(self,agent, epsilon=0):
#         """ Play one episode and collect observations and rewards """

#         summary = dict()
#         summary['rewards'] = list()
#         summary['observations'] = list()
#         summary['actions'] = list()
#         observation = self._transform_observation(self.env.reset())
        
#         t = 0

#         while(True):
#             if(self.render):
#                 self.env.render()
#             action = agent.decide_action(observation, self.env.mask_available_actions(), epsilon)
#             summary['observations'].append(observation)
#             summary['actions'].append(action)
#             observation, reward, done, info = self.env.step(action)
#             observation = self._transform_observation(observation)
#             summary['rewards'].append(reward)

#             if done:
#                 break

#         summary['observations'] = np.vstack(summary['observations'])
#         summary['actions'] = np.vstack(summary['actions'])
#         summary['rewards'] = np.vstack(summary['rewards'])
#         return summary

#     def get_render_info(self):
#         return self.env._scene._render_info
