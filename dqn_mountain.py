from copy import deepcopy
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils
import numpy as np

env = gym.make("MountainCar-v0", render_mode="human")

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer(object):
    def __init__(self, N):
        self.memory = deque([], maxlen=N)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def add_her(self, t, new_g, new_v):
        current_episode_steps = list(self.memory)[-(t+1):]
        for item in current_episode_steps:
            new_state, new_next_state,\
                  new_reward = item.state.clone(), item.next_state.clone(), item.reward.clone() 
            new_state[0][2] =  new_g 
            new_state[0][3] =  new_v 
            new_next_state[0][2] =  new_g 
            new_next_state[0][3] =  new_v
            if new_next_state[0][1].abs() >=  new_v.abs():
                new_reward[0] += 1
            if new_next_state[0][0] == new_g:
                new_next_state = None
                new_reward[0] += 1
                self.push(new_state, item.action.clone(), new_next_state, new_reward)
                break
            self.push(new_state, item.action.clone(), new_next_state, new_reward)


    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_obs, n_action):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(n_obs, 128)
        self.l2 = nn.Linear(128, 254)
        self.l3 = nn.Linear(254, 128)
        self.l4 = nn.Linear(128, n_action)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.l4(x)
    
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 128
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 8000
LR = 1e-4

n_actions = env.action_space.n
state, info = env.reset()
n_obs = len(state) + 2

policy_net = DQN(n_obs, n_actions).to(device)
target_net = DQN(n_obs, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayBuffer(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1 * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], 
                            device=device, dtype=torch.long)
    
episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
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
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, 
                                            batch.next_state)), 
                                            device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_action_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_action_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = reward_batch + (GAMMA * next_state_action_values)

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 60

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = np.append(state, [0.5, 0.06])
    state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
    goal_pos = state[0][0].clone()
    max_val = state[0][1].clone()
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        observation = np.append(observation, [0.5, 0.06])
        if observation[1] >= observation[3]:
            reward += 1
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, device=device, 
                                      dtype=torch.float32).unsqueeze(0)
            
        if goal_pos < state[0][0]:
            goal_pos = state[0][0].clone()
        
        if max_val.abs() < state[0][1].abs():
            max_val = state[0][1].clone()
        
        memory.push(state, action, next_state, reward)

        if done:
            memory.add_her(t, goal_pos, max_val)

        state = next_state

        optimize_model()

        policy_net_state_dict = policy_net.state_dict()
        target_net_state_dict = target_net.state_dict()
        for key in target_net_state_dict:
            target_net_state_dict[key] = TAU * policy_net_state_dict[key] + \
                  (1-TAU) * target_net_state_dict[key]
            
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append((t + 1) * -1)
            plot_durations()
            break

print("complete")
plot_durations(show_result=True)
plt.ioff()
plt.show()

    