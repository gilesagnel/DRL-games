import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
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

SEED = 0

env = gym.make("BreakoutNoFrameskip-v4", render_mode="human", frameskip=1)
env = AtariPreprocessing(env)
env = FrameStack(env, 4) 
env.seed(SEED)

# set up matplotlib
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

    def push(self, state, action, next_state, reward):
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_obs, n_action):
        super(DQN, self).__init__()
        self.c1 = nn.Conv2d(n_obs, 16, kernel_size=8, stride=4)
        self.c2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.c3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.l4 = nn.Linear(1024, 256)
        self.l5 = nn.Linear(256, n_action)

    def forward(self, x):
        out = F.relu(self.c1(x))
        out = F.relu(self.c2(out))
        out = torch.flatten(F.relu(self.c3(out)), start_dim=1)
        out = F.relu(self.l4(out))
        return self.l5(out)
    
def show_image(image):
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Set the title with the image dimensions
    height, width = image.shape
    ax.set_title(f"RGB Image - {width}x{height} pixels")

    # Hide the axes
    ax.axis('off')

    # Add the pixel dimensions to the side
    plt.figtext(0.15, 0.85, f'Width: {width} pixels', fontsize=12, ha='left', va='top')
    plt.figtext(0.15, 0.80, f'Height: {height} pixels', fontsize=12, ha='left', va='top')

    # Show the plot
    plt.show()


GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 32
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 130000
LR = 1e-4

n_actions = env.action_space.n
state, info = env.reset()
n_obs = 4

policy_net = DQN(n_obs, n_actions).to(device)
target_net = DQN(n_obs, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(20000)

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
    
episode_rewards = []


def plot_rewards(show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
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
    nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
    optimizer.step()

def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 8000
else:
    num_episodes = 60

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, device=device, dtype=torch.float32) \
                        .unsqueeze(0)
    total_rewards = 0
    last_states = [state]
    for t in range(10000):
        action = select_action(state)
        observation, reward, terminated, truncated, info = env.step(action.item())
        total_rewards += reward
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, device=device, dtype=torch.float32) \
                                     .unsqueeze(0)
        
        
        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()

        policy_net_state_dict = policy_net.state_dict()
        target_net_state_dict = target_net.state_dict()
        for key in target_net_state_dict:
            target_net_state_dict[key] = TAU * policy_net_state_dict[key] + \
                  (1-TAU) * target_net_state_dict[key]
            
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_rewards.append(total_rewards)
            plot_rewards()
            break
    if (i_episode + 1) % 1000 == 0:
        save_model(policy_net, f"policy_net_episode_{i_episode + 1}.pth")

save_model(policy_net, "policy_net_final.pth")

print("complete")
plot_rewards(show_result=True)
plt.ioff()
plt.show()

    