magicians = ['alice', 'david', 'carolina']
for magician in magicians:
    print(magician)  # 遍历结束是根据缩进来实现

magicians = ['alice', 'david', 'carolina']
print('循环开始')
for magician in magicians:
    print(magician.title() + ", that was a great trick!")
    print(magician.title() + ", that was a great trick!")
print('循环结束--根据缩进结束，没有缩减的只执行一遍')

#####创建数值列表
for value in range(1,5):
    print(value)
numbers = list(range(1, 6))#用range()创建数字列表
print(numbers)

even_numbers = list(range(2,11,2))#使用函数range()时，还可指定步长。例如，下面的代码打印1~10内的偶数：
print(even_numbers)

players = ['charles', 'martina', 'michael', 'florence', 'eli']
print("Here are the first three players on my team:")
for player in players[:3]:#遍历前三个
    print(player.title())

my_foods = ['pizza', 'falafel', 'carrot cake']
friend_foods = my_foods[:]#赋值列表
print("My favorite foods are:")
print(my_foods)
print("\nMy friend's favorite foods are:")
print(friend_foods)

#########################元祖
dimensions = (200, 50)
print(dimensions[0])
print(dimensions[1])
dimensions = (200, 50)
for dimension in dimensions:#遍历元祖
    print(dimension)