with open('pi_digits.txt') as file_object:
    contents = file_object.read()
    print(contents)

with open('E:\project\evun\pythonLearn\learn\chapter10\pi_digits.txt') as file_object:
        contents = file_object.read()
        print(contents)

filename = 'pi_digits.txt'
with open(filename) as file_object:#逐行读出
    for line in file_object:
        print(line)

with open(filename) as file_object:#逐行读出
    for line in file_object:
        print(line.rstrip())

filename = 'pi_digits.txt'
with open(filename) as file_object:
    lines = file_object.readlines()#获取所有的行列表
for line in lines:
    print(line.rstrip())

#######写入文件

filename = 'file_write.txt'
with open(filename, 'w') as file_object:
    file_object.write("I love programming.")
    file_object.write("I love creating new games..\n")
    file_object.write("I love programming.\n")
    file_object.write("I love creating new games.\n")

with open(filename, 'a') as file_object:
    file_object.write("I also love finding meaning in large datasets.\n")
    file_object.write("I love creating apps that can run in a browser.\n")

################################捕捉异常
try:
    print(5/0)
except ZeroDivisionError:
    print("You can't divide by zero!")

filename = 'alice.txt'
try:
    with open(filename) as f_obj:
        contents = f_obj.read()
except FileNotFoundError:
    msg = "Sorry, the file " + filename + " does not exist."
print(msg)
####################存储数据 json.dump()和json.load
import json
numbers = [2, 3, 5, 7, 11, 13]
filename = 'numbers.json'
with open(filename, 'w') as f_obj:
    json.dump(numbers, f_obj)

import json
filename = 'numbers.json'
with open(filename) as f_obj:
    numbers = json.load(f_obj)
print(numbers)