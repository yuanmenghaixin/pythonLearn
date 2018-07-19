class FooBar:
    def __init__(self):
        self.somevar = 42

    def __init__(self, year, name):
        self.somevar = 42
        self.year = year
        self.name = name
        print(name + ":" + year)

    def __init__(self, year):
        self.somevar = 42
        self.year = year
        print("null:" + year)


y = FooBar("1234")
print(y.somevar)


class A:
    def hello(self):
        print("Hello, I'm A.")


class B(A):
    pass


a = A()  # 继承
b = B()
a.hello()
b.hello()


class Bird:
    def __init__(self):
        self.hungry = True

    def eat(self):
        if self.hungry:
            print('Aaaah ...')
            self.hungry = False
        else:
            print('No, thanks!')


class SongBird(Bird):
    def __init__(self):
        Bird.__init__(self)  # 调用未关联的超类构造函数
        self.sound = 'Squawk!'

    def sing(self):
        print(self.sound)


class SongBird(Bird):
    def __init__(self):
        super().__init__()  # 使用函数super
        self.sound = 'Squawk!'

    def sing(self):
        print(self.sound)


class CounterList(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.counter = 0

    def __getitem__(self, index):
        self.counter += 1
        return super(CounterList, self).__getitem__(index)


cl = CounterList(range(10))
print(cl)
cl.reverse()
print(cl)


class Rectangle:
    def __init__(self):
        self.width = 0
        self.height = 0

    def set_size(self, size):
        self.width, self.height = size

    def get_size(self):
        return self.width, self.height


r = Rectangle()
r.width = 10
r.height = 5
print(r.get_size())
r.set_size((150, 100))
print(r.width)


class Rectangle:
    def __init__(self):
        self.width = 0
        self.height = 0

    def set_size(self, size):
        self.width, self.height = size

    def get_size(self):
        return self.width, self.height

    size = property(get_size, set_size)


r = Rectangle()
r.width = 10
r.height = 5
print(r.size)
r.size = 350, 200
print(r.width)
print(r.size)


class MyClass:
    def smeth():  # 静态方法，静态方法参数self
        print('This is a static method')

    smeth = staticmethod(smeth)

    def cmeth(cls):  # 类方法
        print('This is a class method of', cls)

    meth = classmethod(cmeth)


class MyClass:
    @staticmethod  # @列出对应的装饰器
    def smeth():
        print('This is a static method')

    @classmethod  ##@列出对应的装饰器
    def cmeth(cls):
        print('This is a class method of', cls)


#  __getattribute__(self, name)：在属性被访问时自动调用（只适用于新式类）。
#  __getattr__(self, name)：在属性被访问而对象没有这样的属性时自动调用。
#  __setattr__(self, name, value)：试图给属性赋值时自动调用。
#  __delattr__(self, name)：试图删除属性时自动调用。

class Rectangle:
    def __init__(self):
        self.width = 0
        self.height = 0

    def __setattr__(self, name, value):
        if name == 'size':
            self.width, self.height = value
        else:
            self.__dict__[name] = value

    def __getattr__(self, name):
        if name == 'size':
            return self.width, self.height
        else:
            raise AttributeError()

#########迭代器
class Fibs:
    def __init__(self):
        self.a = 0
        self.b = 1

    def __next__(self):#迭代器
        self.a, self.b = self.b, self.a + self.b
        return self.a

    def __iter__(self):
        return self
def flatten(nested):
    for sublist in nested:
        for element in sublist:
            yield element

nested = [[1, 2], [3, 4], [5,6],[7,8]]
for num in flatten(nested):
    print(num)