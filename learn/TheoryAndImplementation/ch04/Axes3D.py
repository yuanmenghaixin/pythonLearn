import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

t = np.arange(-8, 8, 0.25)
X, Y = np.meshgrid(t, t)
R = np.sqrt(X ** 2 + Y ** 2) + np.spacing(1)
Z = np.sin(R) / R

# 显示3维图
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
plt.show()