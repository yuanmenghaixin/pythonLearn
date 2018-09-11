import numpy as np
import matplotlib.pylab as plt

def function_1(x):
    return 0.01*x**2 + 0.1*x

#如下所示，利用微小的差分求导数的过程称为数值微分（numerical
# differentiation）。而基于数学式的推导求导数的过程，则用“解析
# 性”（analytic）一词，称为“解析性求解”或者“解析性求导”。比如，
# y = x2 的导数，可以通过解析性地求解出来。因此，当x = 2时，
# y的导数为4。解析性求导得到的导数是不含误差的“真的导数”。
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

x = np.arange(0.0, 20.0, 0.1) # 以0.1为单位，从0到20的数组x
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))