import matplotlib.pyplot as plt
import figure_test as ft
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

x = [30,40,50,60,70,80,90,100,110,120]
y1 = [157,134,129,123,120,101,83,84,61,43]
y2 = [182,171,154,135,121,106,91,78,68,52]
y3 = [180,169,150,132,119,109,101,88,78,67]
y4 = [218,183,162,142,131,113,106,95,85,77]

bar_width = 0.2
x_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
plt.bar(x_ticks, y1, width=bar_width, label='Random',zorder=10,color='#0072BD',edgecolor='black',linewidth=0.6)
plt.bar([x+bar_width for x in x_ticks], y2, width=bar_width, label='FO',zorder=10,color='#D95319',edgecolor='black',linewidth=0.6)
plt.bar([x+2*bar_width for x in x_ticks], y3, width=bar_width, label='UPSG23',zorder=10,color='#77AC30',edgecolor='black',linewidth=0.6)
plt.bar([x+3*bar_width for x in x_ticks], y4, width=bar_width, label='SDCO',zorder=10,color='#7E2F8E',edgecolor='black',linewidth=0.6)

# 设置横轴标签、标题和图例
plt.xlabel('number of IoTDs')
plt.ylabel('average utility of IoTDs')
plt.xticks([x + bar_width*1.5 for x in x_ticks], x)
plt.legend()
plt.grid(zorder=0)
plt.savefig('../figure/comparision_average_utility.png',dpi=1000, bbox_inches='tight')

plt.show()