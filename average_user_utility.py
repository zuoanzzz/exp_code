from UAV import UAV
from User import User

import random
import math
import matplotlib.pyplot as plt
import numpy as np

w_u = 4   # 信道带宽
noise = 1e-6    # 背景噪声
mu_0 = 100       # 衰减因子
tau = -4        # 信道损失指数
beta = 2      # 复杂系数，每bit任务所需cycles
gamma = 10       # 满意值系数v
theta_M = 1     # 关系函数加权
theta_H = 1     # 奖励-惩罚函数加权
theta_E = 1    # 能耗函数加权
theta_cost = 0.5  # 成本函数加权
iota = 0.5        # 服务器计算能耗系数
theta_uav = 0.12    # 梯度步长
theta_user = 0.02
threshold = 0.001   # 终止阈值

user_number = 60  # 用户数
uav_number = 10

# 单领导者斯坦克尔伯格博弈的实现类
class StackelbergGame:
    def __init__(self, uavs, users):
        self.uavs = uavs
        self.users = users

    # 计算常数部分
    def compute_cons(self, user):
        sr = 0
        s = math.sqrt((user.server.x - user.x)**2 +
                      (user.server.y - user.y)**2+(user.server.z)**2)
        trans_rate = w_u * \
            math.log2(1 + (user.power * mu_0 * s**tau)/noise)    # 噪声功率怎么求？
        for i in range(len(user.server.users)):
            if (user.server.users[i] != user):
                sr += user.relation[i] * gamma *\
                    math.log(1 + user.server.users[i].offload)
        con_uesrs = 0
        for user in user.server.users:
            if(user.offload>0):
                con_uesrs += 1
        frequency = user.server.frequency/con_uesrs
        user.server.con_users = con_uesrs
        A = gamma * (1 + theta_M * sr)
        X = theta_H * (frequency + trans_rate*beta) / \
            (trans_rate * frequency) + \
            theta_E*user.power/trans_rate
        return A, X

    # 计算服务器效用函数梯度
    def uav_grad(self, uav):
        uav_gradient = 0
        con_uesrs = 0
        for user in uav.users:
            if(user.offload>0):
                con_uesrs += 1
        uav.con_users = con_uesrs
        frequency = uav.frequency/uav.con_users
        for user in uav.users:
            A, X = self.compute_cons(user)
            uav_gradient += -1 + A * (X + iota * (frequency**2)
                                      * theta_cost)/(X + theta_cost * uav.price)**2
        return uav_gradient

    # 计算服务器的最终指标
    def uav_utility(self, uav):
        total_offload = 0
        utility = 0
        frequency = uav.frequency/uav.con_users
        for user in uav.users:
            total_offload += user.offload
        utility += (uav.price - iota * (frequency)**2)*user.offload
        energy_consumption = iota * (frequency)**2 * total_offload
        return round(utility, 2), round(energy_consumption, 2), round(total_offload, 2)

    # 计算p_min和p_max
    def compute_p(self, user):
        A, X = self.compute_cons(user)
        p_min = (A/(user.task+1)-X)/theta_cost
        p_max = (A-X)/theta_cost
        return p_min, p_max

    # 计算用户的效用函数梯度
    def user_grad(self, user):
        A, X = self.compute_cons(user)
        user_gradient = A/(1 + user.offload) - X - \
            theta_cost * user.server.price
        return user_gradient

    # 计算用户的最终指标
    def user_utility(self, user):
        F = gamma * math.log(1+user.offload)
        sr = 0
        for i in range(len(user.server.users)):
            if (user.server.users[i] != user):
                sr += user.relation[i] * gamma *\
                    math.log(1 + user.server.users[i].offload)
        M = sr * F
        s = math.sqrt((user.server.x - user.x)**2 +
                      (user.server.y - user.y)**2+(user.server.z)**2)
        trans_rate = w_u * math.log2(1 + (user.power * mu_0 * s**tau)/noise)
        frequency = user.server.frequency/user.server.con_users
        trans_lantency = user.offload/trans_rate
        com_lantency = (user.offload*beta)/frequency
        total_lantency = trans_lantency + com_lantency
        H = user.hopeTime - total_lantency
        energy_consumption = user.power * trans_lantency
        cost = user.server.price * user.offload
        utility = F + theta_M * M + theta_H * H - \
            theta_E * energy_consumption - theta_cost * cost
        return round(utility, 2), round(cost, 2), round(energy_consumption, 2)

    # 迭代算法
    def algorithm(self, max_iters):
        uavs_indexes = []
        users_indexes = []
        for i in range(uav_number):
            uavs_indexes.append([])
            users_indexes.append([])
        for i in range(max_iters):
            for j in range(uav_number):
                uav_gradient = self.uav_grad(self.uavs[j])
                if(abs(uav_gradient) >= threshold):
                    self.uavs[j].price += theta_uav * uav_gradient
                uavs_indexes[j].append(self.uav_utility(self.uavs[j])) 
                for user in self.uavs[j].users:
                    user_gradient = self.user_grad(user)
                    if (abs(user_gradient) >= threshold):
                        p_min, p_max = self.compute_p(user)
                        A,X = self.compute_cons(user)
                        user.offload = A/(X+theta_cost*self.uavs[j].price)-1
                        if (self.uavs[j].price <= p_min):
                            user.offload = user.task
                        elif (self.uavs[j].price >= p_max):
                            user.offload = 0
                    users_indexes[j].append(self.user_utility(user))
        return uavs_indexes, users_indexes
    
    # 平均成本曲线
    def figure_average_user_utility(self,average_user_utility,num_iters):
        plt.plot(np.arange(num_iters),average_user_utility)
        plt.show()

num = 1000
# 创建服务器
uavs = []
for i in range(uav_number):
    uavs.append(UAV(i * 50, i * 50, 50, 1000,frequency=10))
    users = []
    # 创建用户
    for j in range(int(user_number/uav_number)):
        relationship = []
        for k in range(int(user_number/uav_number)):
            relationship.append(random.random())
        relationship[j] = 0
        users.append(User(random.uniform(i * 50 - 25, i * 50 + 25), random.uniform(
            i * 50 - 25, i * 50 + 25), relation=relationship, server=uavs[i]))
    uavs[i].set_users(users)
    uavs[i].set_price(0.1)
# 创建单领导者StackelbergGame的实例
game = StackelbergGame(uavs, users)
uavs_indexes1, users_indexes = game.algorithm(num)
average_user_utility = []
for i in range(num):
    sum_utility = 0
    for user_indexes in users_indexes:
        for j in range(user_number):
            sum_utility += user_indexes[i + j][0]
    average_user_utility.append(sum_utility/user_number)
print(average_user_utility[990:1000])
game.figure_average_user_utility(average_user_utility,num)

uavs = []
for i in range(uav_number):
    uavs.append(UAV(i * 50, i * 50, 50, 1000,frequency=12))
    users = []
    # 创建用户
    for j in range(int(user_number/uav_number)):
        relationship = []
        for k in range(int(user_number/uav_number)):
            relationship.append(random.random())
        relationship[j] = 0
        users.append(User(random.uniform(i * 50 - 25, i * 50 + 25), random.uniform(
            i * 50 - 25, i * 50 + 25), relation=relationship, server=uavs[i]))
    uavs[i].set_users(users)
    uavs[i].set_price(0.1)
# 创建单领导者StackelbergGame的实例
game = StackelbergGame(uavs, users)
uavs_indexes1, users_indexes = game.algorithm(num)
average_user_utility = []
for i in range(num):
    sum_utility = 0
    for user_indexes in users_indexes:
        for j in range(user_number):
            sum_utility += user_indexes[i + j][0]
    average_user_utility.append(sum_utility/user_number)
print(average_user_utility[990:1000])
game.figure_average_user_utility(average_user_utility,num)

uavs = []
for i in range(uav_number):
    uavs.append(UAV(i * 50, i * 50, 50, 1000,frequency=14))
    users = []
    # 创建用户
    for j in range(int(user_number/uav_number)):
        relationship = []
        for k in range(int(user_number/uav_number)):
            relationship.append(random.random())
        relationship[j] = 0
        users.append(User(random.uniform(i * 50 - 25, i * 50 + 25), random.uniform(
            i * 50 - 25, i * 50 + 25), relation=relationship, server=uavs[i]))
    uavs[i].set_users(users)
    uavs[i].set_price(0.1)
# 创建单领导者StackelbergGame的实例
game = StackelbergGame(uavs, users)
uavs_indexes1, users_indexes = game.algorithm(num)
average_user_utility = []
for i in range(num):
    sum_utility = 0
    for user_indexes in users_indexes:
        for j in range(user_number):
            sum_utility += user_indexes[i + j][0]
    average_user_utility.append(sum_utility/user_number)
print(average_user_utility[990:1000])
game.figure_average_user_utility(average_user_utility,num)

uavs = []
for i in range(uav_number):
    uavs.append(UAV(i * 50, i * 50, 50, 1000,frequency=16))
    users = []
    # 创建用户
    for j in range(int(user_number/uav_number)):
        relationship = []
        for k in range(int(user_number/uav_number)):
            relationship.append(random.random())
        relationship[j] = 0
        users.append(User(random.uniform(i * 50 - 25, i * 50 + 25), random.uniform(
            i * 50 - 25, i * 50 + 25), relation=relationship, server=uavs[i]))
    uavs[i].set_users(users)
    uavs[i].set_price(0.1)
# 创建单领导者StackelbergGame的实例
game = StackelbergGame(uavs, users)
uavs_indexes1, users_indexes = game.algorithm(num)
average_user_utility = []
for i in range(num):
    sum_utility = 0
    for user_indexes in users_indexes:
        for j in range(user_number):
            sum_utility += user_indexes[i + j][0]
    average_user_utility.append(sum_utility/user_number)
print(average_user_utility[990:1000])
game.figure_average_user_utility(average_user_utility,num)

# uav_number = 5
# uavs = []
# for i in range(uav_number):
#     uavs.append(UAV(i * 50, i * 50, 50, 1000))
#     users = []
#     # 创建用户
#     for j in range(int(user_number/uav_number)):
#         relationship = []
#         for k in range(int(user_number/uav_number)):
#             relationship.append(random.random())
#         relationship[j] = 0
#         users.append(User(random.uniform(i * 50 - 25, i * 50 + 25), random.uniform(
#             i * 50 - 25, i * 50 + 25), relation=relationship, server=uavs[i]))
#     uavs[i].set_users(users)
#     uavs[i].set_price(0.1)
# # 创建单领导者StackelbergGame的实例
# game = StackelbergGame(uavs, users)
# uavs_indexes1, users_indexes = game.algorithm(num)
# average_user_utility = []
# for i in range(num):
#     sum_utility = 0
#     for user_indexes in users_indexes:
#         for j in range(user_number):
#             sum_utility += user_indexes[i + j][0]
#     average_user_utility.append(sum_utility/user_number)
# print(average_user_utility[290:300])
# game.figure_average_user_utility(average_user_utility,num)

# uav_number = 6
# uavs = []
# for i in range(uav_number):
#     uavs.append(UAV(i * 50, i * 50, 50, 1000))
#     users = []
#     # 创建用户
#     for j in range(int(user_number/uav_number)):
#         relationship = []
#         for k in range(int(user_number/uav_number)):
#             relationship.append(random.random())
#         relationship[j] = 0
#         users.append(User(random.uniform(i * 50 - 25, i * 50 + 25), random.uniform(
#             i * 50 - 25, i * 50 + 25), relation=relationship, server=uavs[i]))
#     uavs[i].set_users(users)
#     uavs[i].set_price(0.1)
# # 创建单领导者StackelbergGame的实例
# game = StackelbergGame(uavs, users)
# uavs_indexes1, users_indexes = game.algorithm(num)
# average_user_utility = []
# for i in range(num):
#     sum_utility = 0
#     for user_indexes in users_indexes:
#         for j in range(user_number):
#             sum_utility += user_indexes[i + j][0]
#     average_user_utility.append(sum_utility/user_number)
# print(average_user_utility[290:300])
# game.figure_average_user_utility(average_user_utility,num)