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
beta = 2.5      # 复杂系数，每bit任务所需cycles
gamma = 10       # 满意值系数v
theta_M = 1     # 关系函数加权
theta_H = 0.5    # 奖励-惩罚函数加权
theta_E = 0.5     # 能耗函数加权
theta_cost = 0.5  # 成本函数加权
iota = 0.5        # 服务器计算能耗系数
theta_uav = 0.03    # 梯度步长
theta_user = 0.03
threshold = 0.001   # 终止阈值

user_number = 15  # 用户数
uav_number = 3

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
        A = gamma * (1 + theta_M * sr)
        X = theta_H * (user.server.frequency + trans_rate*beta) / \
            (trans_rate * user.server.frequency) + \
            theta_E*user.power/trans_rate
        return A, X

    # 计算服务器效用函数梯度
    def uav_grad(self, uav):
        uav_gradient = []
        for k in range(user_number):
            A, X = self.compute_cons(uav.users[k])
            uav_gradient.append(-1 + A * (X + iota * (uav.frequency**2)
                                      * theta_cost)/(X + theta_cost * uav.price[k])**2)
        return uav_gradient

    # 计算服务器的最终指标
    def uav_utility(self, uav):
        total_offload = 0
        utility = 0
        for i in range(user_number):
            total_offload += uav.users[i].offload
            utility += (uav.price[i] - iota * (uav.frequency)**2)*uav.users[i].offload
        energy_consumption = iota * (uav.frequency)**2 * total_offload
        return round(utility, 2), round(energy_consumption, 2)

    # 计算p_min和p_max
    def compute_p(self, user):
        A, X = self.compute_cons(user)
        p_min = (A/(user.task+1)-X)/theta_cost
        p_max = (A-X)/theta_cost
        return p_min, p_max

    # 计算用户的效用函数梯度
    def user_grad(self, user, i):
        A, X = self.compute_cons(user)
        user_gradient = A/(1 + user.offload) - X - \
            theta_cost * user.server.price[i]
        return user_gradient

    # 计算用户的最终指标
    def user_utility(self, user, i):
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
        trans_lantency = user.offload/trans_rate
        com_lantency = (user.offload*beta)/user.server.frequency
        total_lantency = trans_lantency + com_lantency
        H = user.hopeTime - total_lantency
        energy_consumption = user.power * trans_lantency
        cost = user.server.price[i] * user.offload
        utility = F + theta_M * M + theta_H * H - \
            theta_E * energy_consumption - theta_cost * cost
        return round(utility, 2), round(total_lantency, 2), round(energy_consumption, 2)

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
                for k in range(user_number):
                    if(abs(uav_gradient[k]) >= threshold):
                        self.uavs[j].price[k] += theta_uav * uav_gradient[k]
                    uavs_indexes[j].append(self.uav_utility(self.uavs[j])) 
                for k in range(user_number):
                    user_gradient = self.user_grad(self.uavs[j].users[k],k)
                    if (abs(user_gradient) >= threshold):
                        p_min, p_max = self.compute_p(self.uavs[j].users[k])
                        A,X = self.compute_cons(self.uavs[j].users[k])
                        self.uavs[j].users[k].offload = theta_user * user_gradient
                        if (self.uavs[j].price[k] <= p_min):
                            self.uavs[j].users[k].offload = self.uavs[j].users[k].task
                        elif (self.uavs[j].price[k] >= p_max):
                            self.uavs[j].users[k].offload = 0
                    users_indexes[j].append(self.user_utility(self.uavs[j].users[k], k))
        return uavs_indexes, users_indexes
    
    # 无人机效用函数图
    def figure_uav_utility(self,uav_utility,num_converge):
        plt.plot(num_converge, uav_utility, 
            marker='x', linewidth=0.5, color='blue', label='uav_utility')
        plt.legend()
        plt.xlabel("number of iterations", fontsize=14)
        plt.ylabel("utility", fontsize=14)
        plt.show()
    
    # 用户效用函数图
    def figure_user_utility(self,user_utility,num_converge):
        plt.plot(num_converge, user_utility,
                    marker='x', linewidth=0.5, color='red', label='user_utility')
        plt.legend()
        plt.xlabel("number of iterations", fontsize=14)
        plt.ylabel("utility", fontsize=14)
        plt.show()

# 创建服务器
uavs = []
for i in range(uav_number):
    uavs.append(UAV(i * 50, i * 50, 50, 1000))
    users = []
    price = []
    # 创建用户
    for j in range(user_number):
        relationship = []
        for k in range(user_number):
            relationship.append(random.random())
        relationship[j] = 0
        users.append(User(random.uniform(i * 50 - 25, i * 50 + 25), random.uniform(
            i * 50 - 25, i * 50 + 25), relation=relationship, server=uavs[i]))
        price.append(0.1)
    uavs[i].set_users(users)
    uavs[i].set_price(price)

# 创建单领导者StackelbergGame的实例
num = 400
game = StackelbergGame(uavs, users)
uavs_indexes, users_indexes = game.algorithm(num)

# uav_utility = []
# for i in range(0,len(uavs_indexes[0]),2):
#     sum_utility = 0
#     for uav_indexes in uavs_indexes:
#         sum_utility += uav_indexes[i][0]
#     uav_utility.append(sum_utility)
user_utility = []
for i in range(int(len(users_indexes[0])/user_number)):
    sum_utility = 0
    for user_indexes in users_indexes:
        for j in range(user_number):
            sum_utility += user_indexes[i + j][0]
    user_utility.append(sum_utility/(user_number*uav_number))

#game.figure_uav_utility(uav_utility,np.arange(1, len(uavs_indexes[0]) + 1,2))
game.figure_user_utility(user_utility,np.arange(1, len(users_indexes[0])/user_number + 1))