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
theta_M = 0.5     # 关系函数加权
theta_H = 1     # 奖励-惩罚函数加权
theta_E = 1.5     # 能耗函数加权
theta_cost = 0.5  # 成本函数加权
iota = 2.0        # 服务器计算能耗系数
theta_uav = 0.128    # 梯度步长
theta_user = 0.02
threshold = 0.001   # 终止阈值

user_number = 40  # 用户数
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
        return round(utility, 2), round(energy_consumption, 2)

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
    
    # 无人机效用函数图
    def figure_uav_utility(self,uav_utility,num_iters):
        # plt.plot(np.arange(0,len(uavs_indexes[0]),2), uav_utility, 
        #      linewidth=1, color='blue', label='uav_utility')
        

        # 绘制完整曲线
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(np.arange(0,num_iters), uav_utility, 
             linewidth=1, color='blue', label='uav_utility')
        ax.plot(230, round(uav_utility[230],2), marker='o', markersize=3, color='black')
        ax.plot(280, round(uav_utility[280],2), marker='o', markersize=3, color='black')
        ax.annotate('({},{})'.format(230, round(uav_utility[230],2)), xy=(230, round(uav_utility[230],2)), xytext=(-90, -40), textcoords='offset points',color = 'red', fontweight='bold', arrowprops=dict(arrowstyle='->'))
        ax.annotate('({},{})'.format(280, round(uav_utility[280],2)), xy=(280, round(uav_utility[280],2)), xytext=(-60, -40), textcoords='offset points',color = 'red', fontweight='bold', arrowprops=dict(arrowstyle='->'))
        ax.set_xlabel("number of iterations", fontsize=14)
        ax.set_ylabel("utility", fontsize=14)
        ax.legend()
        ax.set_xticks(np.arange(0,len(uav_utility)+1,50))
        ax.set_xlim(0,300)

        # 绘制放大曲线
        sub_x = np.arange(150,251)
        sub_y = uav_utility[149:250]
        # sub_fig, sub_ax = plt.subplots(figsize=(6, 4))
        # sub_ax.plot(sub_x, sub_y, label='uav_utility')
        # sub_ax.set_xlabel('number of iterations')
        # sub_ax.set_ylabel('utility')

        # 将放大曲线插入到大图中
        axins = ax.inset_axes([0.3, 0.2, 0.5, 0.5])
        axins.plot(sub_x, sub_y, linewidth=1)
        axins.set_xlim(150, 250)
        axins.set_xticks(np.arange(150,251,20))
        #axins.set_ylim(uav_utility[149])
        ax.indicate_inset_zoom(axins)
        #plt.savefig('./figure/uav_utility.png',dpi=1000, bbox_inches='tight')
        plt.show()

# 创建服务器
uavs = []
for i in range(uav_number):
    uavs.append(UAV(i * 50, i * 50, 50, 1000))
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
temp_users = users
temp_uavs = uavs

# 创建单领导者StackelbergGame的实例
# iota = 1.5
num = 300
game = StackelbergGame(temp_uavs, temp_users)
uavs_indexes1, users_indexes = game.algorithm(num)
uav_utility = []
for i in range(1,len(uavs_indexes1[0])):
    sum_utility = 0
    for uav_indexes in uavs_indexes1:
        sum_utility += uav_indexes[i][0]
    uav_utility.append(sum_utility)
print(uav_utility[290:300])
game.figure_uav_utility(uav_utility,len(uavs_indexes1[0])-1)

# temp_users = users
# temp_uavs = uavs
# iota = 1.0
# game = StackelbergGame(temp_uavs, temp_users)
# uavs_indexes2, users_indexes = game.algorithm(num)
# uav_utility = []
# for i in range(1,len(uavs_indexes2[0])):
#     sum_utility = 0
#     for uav_indexes in uavs_indexes2:
#         sum_utility += uav_indexes[i][0]
#     uav_utility.append(sum_utility)
# print(uav_utility[290:300])
# game.figure_uav_utility(uav_utility,len(uavs_indexes2[0])-1)

# iota = 0.5
# temp_users = users
# temp_uavs = uavs
# game = StackelbergGame(temp_uavs, temp_users)
# uavs_indexes3, users_indexes = game.algorithm(num)
# uav_utility = []
# for i in range(len(uavs_indexes3[0])):
#     sum_utility = 0
#     for uav_indexes in uavs_indexes3:
#         sum_utility += uav_indexes[i][0]
#     uav_utility.append(sum_utility)
# uav_utility.append(uav_utility[299])
# print(uav_utility[290:300])
# game.figure_uav_utility(uav_utility,len(uavs_indexes3[0])+1)