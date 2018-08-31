import numpy as np
a=np.array([[1,2],[3,4]])
b=np.array([[4,3],[2,1]])

print(a*b)
print(np.dot(a,b))
print(np.matmul(a,b))#矢量积
print(np.multiply(a,b))#数量积

# c=np.array([[1,1,1],[0,1,1],[0,0,1]])
# d=np.array([[1,0,1],[0,1,0],[1,0,1]])
#
# print(c*d)
# print(np.dot(c,d))
# print(np.matmul(c,d))#矢量积
# print(np.multiply(c,d))#数量积