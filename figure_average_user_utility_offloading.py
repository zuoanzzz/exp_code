import matplotlib.pyplot as plt

x = [1,2,3,4,5,6,7,8,9,10]
y1 = [74, 88, 104, 133, 146, 163, 195, 232, 276, 317]
y2 = [74, 137, 165, 198, 238, 286, 324, 384, 468, 559]
y3 = [74, 147, 225, 308, 372, 449, 542, 654, 779, 881]
y4 = [74, 157, 295, 388, 528, 639, 805, 922, 1039, 1205]

plt.plot(x,y1,marker='s',linewidth=1,label='number of computations = 15MB',markersize=5, linestyle='--')
plt.plot(x,y2,marker='^',linewidth=1,label='number of computations = 30MB',markersize=5, linestyle='--')
plt.plot(x,y3,marker='D',linewidth=1,label='number of computations = 45MB',markersize=5, linestyle='--')
plt.plot(x,y4,marker='p',linewidth=1,label='number of computations = 60MB',markersize=5, linestyle='--')
plt.legend()
plt.xlabel("number of UAVs")
plt.ylabel("average utility of IoTDs")
plt.xticks(x)
plt.grid()
plt.savefig('../figure/average_user_utility_offloading.png',dpi=1000, bbox_inches='tight')

plt.show()