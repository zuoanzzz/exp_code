from testDo import UAV, User

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

w_u = 4   # 信道带宽
noise = 1e-6    # 背景噪声
mu_0 = 100       # 衰减因子
tau = -4        # 信道损失指数
beta = 2      # 复杂系数，每bit任务所需cycles
gamma = 10       # 满意值系数v
theta_M = 0.5     # 关系函数加权
theta_H = 1     # 奖励-惩罚函数加权
theta_E = 1.5     # 能耗函数加权
theta_cost = 0.5  # 成本函数加权
iota = 1        # 服务器计算能耗系数
theta_uav = 0.128    # 梯度步长
theta_user = 0.02
threshold = 0.001   # 终止阈值

# Actor Network
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Actions are typically within [-1, 1]

# Critic Network
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = ActorNetwork(state_dim, action_dim)
        self.actor_target = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim, action_dim)
        self.critic_target = CriticNetwork(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)
        
        # Compute target Q value
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q_value = reward + self.gamma * (1 - done) * self.critic_target(next_state, next_action)
        
        # Update Critic
        q_value = self.critic(state, action)
        critic_loss = nn.MSELoss()(q_value, target_q_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        action = self.actor(state)
        actor_loss = -self.critic(state, action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self._update_target(self.actor_target, self.actor)
        self._update_target(self.critic_target, self.critic)
    
    def _update_target(self, target_network, source_network):
        for target_param, param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.actor(state).detach().numpy().flatten()
    
    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

# 修改环境类
class UAVEnvironment:
    def __init__(self, uavs, users):
        self.uavs = uavs
        self.users = users
        self.agent = DDPGAgent(state_dim=len(users)*3 + len(uavs)*3, action_dim=len(users))
        self.state = self._get_initial_state()

    def _get_initial_state(self):
        state = []
        for user in self.users:
            state.extend([user.x, user.y, user.task])
        for uav in self.uavs:
            state.extend([uav.x, uav.y, uav.frequency])
        return state

    def reset(self):
        self.state = self._get_initial_state()
        return self.state

    def step(self, action):
        # 执行动作，返回下一个状态、奖励和是否终止
        for i, user in enumerate(self.users):
            user.offload = action[i]
        
        next_state = self._get_initial_state()
        reward = self._compute_reward()
        done = self._check_done()
        
        return next_state, reward, done

    def _compute_reward(self):
        reward = 0
        for user in self.users:
            utility, latency, energy_consumption = self._compute_user_utility(user)
            reward += utility - latency - energy_consumption
        return reward

    def _compute_user_utility(self, user):
        F = gamma * np.log(1 + user.offload)
        sr = sum(user.relation[i] * gamma * np.log(1 + other_user.offload)
                 for i, other_user in enumerate(user.server.users) if other_user != user)
        M = sr * F
        s = np.sqrt((user.server.x - user.x)**2 + (user.server.y - user.y)**2 + (user.server.z)**2)
        trans_rate = w_u * np.log2(1 + (user.power * mu_0 * s**tau) / noise)
        frequency = user.server.frequency / len(user.server.con_users)
        trans_lantency = user.offload / trans_rate
        com_lantency = (user.offload * beta) / frequency
        total_lantency = trans_lantency + com_lantency
        H = user.hopeTime - total_lantency
        energy_consumption = user.power * trans_lantency
        cost = user.server.price * user.offload
        utility = F + theta_M * M + theta_H * H - theta_E * energy_consumption - theta_cost * cost
        return round(utility, 2), round(total_lantency, 2), round(energy_consumption, 2)

    def _check_done(self):
        return False  # 在训练期间，我们可能希望环境永不结束

    def run_episode(self):
        state = self.reset()
        done = False
        while not done:
            action = self.agent.select_action(state)
            next_state, reward, done = self.step(action)
            self.agent.store(state, action, reward, next_state, done)
            self.agent.update()
            state = next_state


# 创建 UAV 和用户
uavs = [UAV(x * 50, y * 50, 50, 1000) for x in range(3) for y in range(3)]
users = [User(x * 10, y * 10, task=random.randint(15, 30)) for x in range(10) for y in range(6)]

# 为每个用户分配一个 UAV
for i, user in enumerate(users):
    user.server = uavs[i % len(uavs)]
    uavs[i % len(uavs)].con_users.append(user)

# 训练环境
env = UAVEnvironment(uavs, users)
for episode in range(300):
    env.run_episode()
