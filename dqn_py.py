import math 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-white')
import pandas as pd
from matplotlib import animation, rc
plt.rcParams.update({'pdf.fonttype': 'truetype'})



## Environment simulator
def plus(x):
    return 0 if x < 0 else x
def minus(x):
    return 0 if x > 0 else -x
def shock(x): #웹페이지에서 봤던 충격함수.(충격함수는 가격 변화에 따른 수요영향력)
    y = np.sqrt(0.0000007*(x**4)-0.00005*(x**3)-0.0036*(x**2)-0.0106*x+5.3552 )
    y_2 = y /120
    
    return y_2

#Demand at time step t for current price p_t and previous price p_t_1
def q_t(p_t, p_t_1, q_0, k, a, b, q_setting):
    #print(f'q_set: {q_setting}, p_t: {p_t}, demand:{q_0[q_setting]}')
    return plus(q_0[q_setting] - k*p_t - a*shock(plus(p_t - p_t_1)) + b*shock(minus(p_t - p_t_1))) #웹페이지에서 봤던 수요함수임(d(t,j))

def profit_t(p_t, p_t_1, q_0, k, a, b, unit_cost, q_setting):
    return q_t(p_t, p_t_1, q_0, k, a, b, q_setting)*(p_t - unit_cost) #q_t(,,,) 이거는 demand임. #(p_t - unit_cost)는 순이익임(가격-유닛당 비용) 즉, 리턴값(profit) = 수요 * 순이익
# Total profit for price vector p over len(p) time steps
def profit_total(p, unit_cost, q_0, k, a, b):
    # 여기 0넣으면 안될듯 아마
    return profit_t(p[0], p[0], q_0, k, 0, 0, unit_cost, 0) + sum(map(lambda t: profit_t(p[t], p[t-1], q_0, k, a, b, unit_cost, t), range(len(p))))


def profit_total_dqn(p, unit_cost, q_0, k, a, b):
    # 여기 0넣으면 안될듯 아마
    #print(sum(map(lambda t: profit_t(p[t], p[t-1], q_0, k, a, b, unit_cost, t), range(len(p)))))
    return profit_t(p[0], p[0], q_0, k, 0, 0, unit_cost, 0) + sum(map(lambda t: profit_t(p[0], p[1], q_0, k, a, b, unit_cost, t), range(7)))

## Environment parameters
T = 7
price_max = 2500
price_step = 200
q_0 = [  6000, 22815, 10000,24000, 4000, 14206, 8009]
p_minus = 2100
k = 4.42
unit_cost = 1000
a_q = 60
b_q = 20

## Partial bindings for readability
def profit_t_response(p_t, p_t_1, q_setting):
    return profit_t(p_t, p_t_1, q_0, k, a_q, b_q, unit_cost, q_setting) #택배 도메인에 맞춰서 들어가야 하는 상수.

def profit_response(p):
    return profit_total(p, unit_cost, q_0, k, a_q, b_q)

# def profit_response_dqn(p):
#     return profit_total_dqn(p, unit_cost, q_0, k, a_q, b_q)
def profit_response_dqn(p_t, p_t_1, t):
    return profit_t(p_t, p_t_1, q_0, k, a_q, b_q, unit_cost, t)

price_grid = np.arange(1500, price_max, price_step) ## price_step: 최소가격, 뒤 price_step: 가격 간격
price_change_grid = np.arange(0.5, 2.0, 0.1) ## [0.5, 0.6, 0.7 ,,, 1.9]
profit_map = np.zeros( (len(price_grid), len(price_change_grid)) ) #행 : price_gird 개수, 열 : price_change_grid 개수


import math
import random
import numpy as np
from IPython.display import clear_output
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# A cyclic buffer of bounded size that holds the transitions observed recently
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class PolicyNetworkDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetworkDQN, self).__init__()
        layers = [
              nn.Linear(state_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, action_size)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        q_values = self.model(x)
        return q_values  

class AnnealedEpsGreedyPolicy(object):
    def __init__(self, eps_start = 0.9, eps_end = 0.05, eps_decay = 400):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

    def select_action(self, q_values):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            return np.argmax(q_values)
        else:
            return random.randrange(len(q_values))

GAMMA = 1.10
TARGET_UPDATE = 20
BATCH_SIZE = 512

def update_model(memory, policy_net, target_net):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.stack(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = reward_batch[:, 0] + (GAMMA * next_state_values)  

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def env_intial_state():
    return np.repeat(0, 2*T)

def env_step(t, state, action):
    next_state = np.repeat(0, len(state))
    next_state[0] = price_grid[action]
    next_state[1:T] = state[0:T-1]
    next_state[T+t] = 1
    # reward = profit_t_response(next_state[t-1], next_state[1], t)
    # reward = profit_response_dqn([next_state[0], next_state[1]])
    if t != 0:
        reward = profit_t_response(next_state[t-1], next_state[t], t)
    else:
        reward = profit_t_response(p_minus, next_state[0], t)
        
    return next_state, reward

def to_tensor(x):
    return torch.from_numpy(np.array(x).astype(np.float32))

def to_tensor_long(x):
    return torch.tensor([[x]], device=device, dtype=torch.long)

policy_net = PolicyNetworkDQN(2*T, len(price_grid)).to(device)
target_net = PolicyNetworkDQN(2*T, len(price_grid)).to(device)
optimizer = optim.AdamW(policy_net.parameters(), lr = 0.001)
policy = AnnealedEpsGreedyPolicy()
memory = ReplayMemory(10000)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

num_episodes = 1000
return_trace = []




p_trace = [] # price schedules used in each episode
for i_episode in range(num_episodes):
    state = env_intial_state()
    reward_trace = []
    p = []
    for t in range(T):
        # Select and perform an action
        with torch.no_grad():
            q_values = policy_net(to_tensor(state))
        action = policy.select_action(q_values.detach().numpy())

        next_state, reward = env_step(t, state, action)

        # Store the transition in memory
        memory.push(to_tensor(state), 
                    to_tensor_long(action), 
                    to_tensor(next_state) if t != T - 1 else None, 
                    to_tensor([reward]))

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        update_model(memory, policy_net, target_net)

        reward_trace.append(reward)
        p.append(price_grid[action])
    



    return_trace.append(sum(reward_trace))
    p_trace.append(p)

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

        clear_output(wait = True)
        print(f'Episode {i_episode} of {num_episodes} ({i_episode/num_episodes*100:.2f}%)')





for profit in sorted(profit_response(s) for s in p_trace)[-10:]:
    print(f'Best profit results: {profit}')