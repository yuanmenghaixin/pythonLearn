import numpy as np
#参数y和t是NumPy数组。函数内部在计算np.log时，加上了一
#个微小值delta。这是因为，当出现np.log(0)时，np.log(0)会变为负无限大
# 的-inf，这样一来就会导致后续计算无法进行。作为保护性对策，添加一个
# 微小值可以防止负无限大的发生。下面，我们使用cross_entropy_error(y, t)
# 进行一些简单的计算。
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))
