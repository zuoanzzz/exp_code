import matplotlib.pyplot as plt

x = [30,40,50,60,70,80,90,100,110,120]
y1 = [2013,2789,3731,4095,3716,4445,4963,5111,6356,6456]
y2 = [3010,3826,4021,5123,5881,6267,7110,7553,7861,8163]
y3 = [2946,3581,3944,4798,5613,6581,7438,7946,8512,9430]
y4 = [3143.1,4022.67,4341.29,5619.99,6113.16,7171.66,8088.79,8770.51,9669.27,10689.88]

bar_width = 0.2
x_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
plt.bar(x_ticks, y1, width=bar_width, label='Random',zorder=10,color='#0072BD',edgecolor='black',linewidth=0.6)
plt.bar([x+bar_width for x in x_ticks], y2, width=bar_width, label='FO',zorder=10,color='#D95319',edgecolor='black',linewidth=0.6)
plt.bar([x+2*bar_width for x in x_ticks], y3, width=bar_width, label='UPSG23',zorder=10,color='#77AC30',edgecolor='black',linewidth=0.6)
plt.bar([x+3*bar_width for x in x_ticks], y4, width=bar_width, label='SDCO',zorder=10,color='#7E2F8E',edgecolor='black',linewidth=0.6)

# 设置横轴标签、标题和图例
plt.xlabel('number of IoTDs')
plt.ylabel('overall utility of UAVs')
plt.xticks([x + bar_width*1.5 for x in x_ticks], x)
plt.legend()
plt.grid(zorder=0)
plt.savefig('../figure/comparision_uav_utility.png',dpi=1000, bbox_inches='tight')

plt.show()