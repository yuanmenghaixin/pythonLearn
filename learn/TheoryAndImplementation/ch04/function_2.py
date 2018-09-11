import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

def function_2(x,y):
    return x ** 2 + y ** 2
    # 或者return np.sum(x**2)

t = np.arange(-3, 3, 0.1)
X, Y = np.meshgrid(t, t)
Z = function_2(X,Y)

# 显示3维图
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
ax.set_xlabel('x label', color='r')
ax.set_ylabel('y label', color='g')
ax.set_zlabel('z label', color='b')#给三个坐标轴注明
plt.show()