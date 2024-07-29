import matplotlib.pyplot as plt
import figure_test as ft
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

x = [30,40,50,60,70,80,90,100,110,120]
y1 = [2985.92,3783.36,4232.44,5502.16,6056.32,7098.27,8033.74,8725.11,9630.4,10658.24]
y2 = [3143.1,4022.67,4341.29,5619.99,6113.16,7171.66,8088.79,8770.51,9669.27,10689.88]
y3 = [3235.15,4239,4443.24,5734.45,6165.94,7243.74,8143.1,8815.37,9707.68,10721.14]
y4 = [3632.62,4431.37,4794.83,5967.84,6404.75,7313.45,8185.68,8992.16,9718.76,10787.63]

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
plt.bar(x_ticks, y4, width=bar_width, label=r"$\iota=0.5$",zorder=10,color='#0072BD',edgecolor='black',linewidth=0.6)
plt.bar([x+bar_width for x in x_ticks], y3, width=bar_width, label=r"$\iota=1.0$",zorder=10,color='#D95319',edgecolor='black',linewidth=0.6)
plt.bar([x+2*bar_width for x in x_ticks], y2, width=bar_width, label=r"$\iota=1.5$",zorder=10,color='#7E2F8E',edgecolor='black',linewidth=0.6)
plt.bar([x+3*bar_width for x in x_ticks], y1, width=bar_width, label=r"$\iota=2.0$",zorder=10,color='#77AC30',edgecolor='black',linewidth=0.6)

# 设置横轴标签、标题和图例
plt.xlabel('number of IoTDs')
plt.ylabel('overall utility of UAVs')
plt.xticks([x + bar_width*1.5 for x in x_ticks], x)
plt.legend()
plt.grid(zorder=0)
plt.savefig('../figure/bar_uav_energy_consumption.png',dpi=1000, bbox_inches='tight')

plt.show()