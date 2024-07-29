import matplotlib.pyplot as plt
import figure_test as ft
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

x = [1,2,3,4,5,6,7,8,9,10]
y1 = [9702.07,14318.3,19016.36,23188.92,26283.1,28902.83,29668.47,30189.23,30403.41,30674.74]
y2 = [10014.37,14956.2,19951.2,23275.19,26430.51,29430.52,30085.92,30593.85,30679.26,31213.99]
y3 = [10379.59,15146.85,20589.64,23410.46,26810.46,29712.66,30487.05,30802.78,31300.94,31781.92]
y4 = [10516.99,15450.65,20621.89,23739.9,27011.8,29996.69,30763.17,31348.21,31750.28,32112.12]

# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(x,y4,marker='d',linewidth=0.8,label=r"$\iota=0.5$",markerfacecolor='none',markersize=5,color='crimson')
# ax.plot(x,y3,marker='D',linewidth=0.8,label=r"$\iota=1.0$",markerfacecolor='none',markersize=5)
# ax.plot(x,y2,marker='^',linewidth=0.8,label=r"$\iota=1.5$",markerfacecolor='none',markersize=5)
# ax.plot(x,y1,marker='o',linewidth=0.8,label=r"$\iota=2.0$",markerfacecolor='none',markersize=5)
# ax.legend()
# ax.set_xlabel("number of users")
# ax.set_ylabel("utility of uavs")
# # plt.xlim(30,120)
# # plt.ylim(2900)
# ax.set_xticks(x)
# ax.grid()


# # 绘制放大曲线
# sub_x = [110,120]
# sub_y1 = y1[8:10]
# sub_y2 = y2[8:10]
# sub_y3 = y3[8:10]
# sub_y4 = y4[8:10]

# # 绘制缩放图
# axins = ax.inset_axes((0.6, 0.05, 0.4, 0.4))

# # 在缩放图中也绘制主图所有内容，然后根据限制横纵坐标来达成局部显示的目的
# axins.plot(x,y4,marker='d',linewidth=0.8,label=r"$\iota=0.5$",markerfacecolor='none',markersize=5,color='crimson')
# axins.plot(x,y3,marker='D',linewidth=0.8,label=r"$\iota=1.0$",markerfacecolor='none',markersize=5)
# axins.plot(x,y2,marker='^',linewidth=0.8,label=r"$\iota=1.5$",markerfacecolor='none',markersize=5)
# axins.plot(x,y1,marker='o',linewidth=0.8,label=r"$\iota=2.0$",markerfacecolor='none',markersize=5)

# # 局部显示并且进行连线
# ft.zone_and_linked(ax, axins, 8, 9, x , [y4,y3,y2,y1], 'bottom')

# plt.show()


# # 将放大曲线插入到大图中
# axins = inset_axes(ax, width="40%", height="30%",loc='lower left',
#                    bbox_to_anchor=(0.5, 0.1, 1, 1),
#                    bbox_transform=ax.transAxes)
# axins.plot(sub_x, sub_y4, linewidth=0.8)
# axins.plot(sub_x, sub_y3, linewidth=0.8)
# axins.plot(sub_x, sub_y2, linewidth=0.8)
# axins.plot(sub_x, sub_y1, linewidth=0.8)
# axins.set_xticks(sub_x)
# ax.indicate_inset_zoom(axins)
# mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

# plt.savefig('./figure/uav_energy_consumption.png',dpi=1000, bbox_inches='tight')
# plt.show()

# 使用bar函数绘制柱状图
bar_width = 0.2
x_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
plt.bar(x_ticks, y1, width=bar_width, label=r"$f_u=10$",zorder=10,color='#0072BD',edgecolor='black',linewidth=0.6)
plt.bar([x+bar_width for x in x_ticks], y2, width=bar_width, label=r"$f_u=12$",zorder=10,color='#D95319',edgecolor='black',linewidth=0.6)
plt.bar([x+2*bar_width for x in x_ticks], y3, width=bar_width, label=r"$f_u=14$",zorder=10,color='#7E2F8E',edgecolor='black',linewidth=0.6)
plt.bar([x+3*bar_width for x in x_ticks], y4, width=bar_width, label=r"$f_u=16$",zorder=10,color='#77AC30',edgecolor='black',linewidth=0.6)

# 设置横轴标签、标题和图例
plt.xlabel('number of UAVs')
plt.ylabel('overall utility of UAVs')
plt.xticks([x + bar_width*1.5 for x in x_ticks], x)
plt.legend()
plt.grid(zorder=0)
plt.savefig('../figure/bar_total_utility_frequency.png',dpi=1000, bbox_inches='tight')

plt.show()