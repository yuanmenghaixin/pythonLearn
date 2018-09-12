# coding: utf-8

# 乘法层
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    # 正向传播
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    # 反向传播 dout 对应的导数
    #。backward()将从上游传来的导数（dout）乘以正向传播的翻转值，然后传给下游
    def backward(self, dout):
        dx = dout * self.y # 翻转x和y
        dy = dout * self.x

        return dx, dy


# 加法层
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy
