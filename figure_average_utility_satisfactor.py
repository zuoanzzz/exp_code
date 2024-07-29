import matplotlib.pyplot as plt

x = [30,40,50,60,70,80,90,100,110,120]
y4 = [479, 416, 376, 327, 283, 244, 214, 188, 165, 155]
y3 = [343, 306, 272, 231, 197, 166, 138, 121, 105, 98]
y2 = [218, 183, 162, 142, 131, 113, 106, 95, 85, 77]
y1 = [99, 91, 80, 72, 65, 58, 53, 47, 43, 38]

plt.plot(x,y1,marker='o',linewidth=1,label='satisfactor'r"$(\gamma)$"' = 5',markersize=5, linestyle='--')
plt.plot(x,y2,marker='^',linewidth=1,label='satisfactor'r"$(\gamma)$"' = 10',markersize=5, linestyle='--')
plt.plot(x,y3,marker='D',linewidth=1,label='satisfactor'r"$(\gamma)$"' = 15',markersize=5, linestyle='--')
plt.plot(x,y4,marker='p',linewidth=1,label='satisfactor'r"$(\gamma)$"' = 20',markersize=5, linestyle='--')
plt.legend()
plt.xlabel("number of IoTDs")
plt.ylabel("average utility of IoTDs")
plt.xticks(x)
plt.grid()
plt.savefig('./figure/average_user_utility_gamma.png',dpi=1000, bbox_inches='tight')

plt.show()