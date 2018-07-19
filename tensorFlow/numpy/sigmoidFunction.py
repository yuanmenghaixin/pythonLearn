import numpy as np
#自定义sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print(sigmoid(100))
print(sigmoid(0))
print(sigmoid(-1000))
def initilize_with_zeros(dim):
    w = np.zeros((dim, 1)) #zeros创建0矩阵
    b = 0.0
    #assert(w.shape == (dim, 1))
    #assert(isinstance(b, float) or isinstance(b, int))
    return w, b

print(np.log(np.e))#等于1

e= np.arange(9)
print(e)
a = e.reshape(3,3,1) #数组对象中的方法，用于改变数组的形状。
print(a)
print(np.squeeze(a))

#自定义传播函数
def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)#dot()矩阵的乘积
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m
    assert (dw.shape == w.shape)
    assert (db.dtype == float)

    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {'dw': dw, 'db': db}

    return grads, cost