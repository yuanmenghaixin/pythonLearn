##########函数
def greet_user():  # 函数
    print("Hello!")
greet_user()


def greet_user(username):  # 带参数的函数
    print("Hello, " + username.title() + "!")


greet_user('jesse')


def describe_pet(animal_type, pet_name):
    print("\nI have a " + animal_type + ".")
    print("My " + animal_type + "'s name is " + pet_name.title() + ".")


describe_pet('hamster', 'harry')
describe_pet('dog', 'willie')


def describe_pet(pet_name, animal_type):  # 关键字实参
    print("\nI have a " + animal_type + ".")
    print("My " + animal_type + "'s name is " + pet_name.title() + ".")


describe_pet(animal_type='hamster', pet_name='harry')


def describe_pet(pet_name, animal_type='dog'):  # 函数参数默认值
    print("\nI have a " + animal_type + ".")
    print("My " + animal_type + "'s name is " + pet_name.title() + ".")


describe_pet(pet_name='willie')


def get_formatted_name(first_name, last_name):  # 返回值函数
    full_name = first_name + ' ' + last_name
    return full_name.title()


musician = get_formatted_name('jimi', 'hendrix')
print(musician)


def get_formatted_name(first_name, last_name, middle_name=''):
    if middle_name:  # 字符串为空判断
        full_name = first_name + ' ' + middle_name + ' ' + last_name
    else:
        full_name = first_name + ' ' + last_name
    return full_name.title()


musician = get_formatted_name('jimi', 'hendrix')
print(musician)
musician = get_formatted_name('john', 'hooker', 'lee')
print(musician)


def build_person(first_name, last_name):  # 返回字典
    person = {'first': first_name, 'last': last_name}
    return person


musician = build_person('jimi', 'hendrix')
print(musician)


def greet_users(names):  # 参数为传递列表
    for name in names:
        msg = "Hello, " + name.title() + "!"
        print(msg)


usernames = ['hannah', 'ty', 'margot']
greet_users(usernames)


def print_models(unprinted_designs, completed_models):
    while unprinted_designs:
        current_design = unprinted_designs.pop()
        print("Printing model: " + current_design)
        completed_models.append(current_design)


def show_completed_models(completed_models):
    print("\nThe following models have been printed:")
    for completed_model in completed_models:
        print(completed_model)


unprinted_designs = ['iphone case', 'robot pendant', 'dodecahedron']
completed_models = []
# 传递副本
print("传递副本");
print_models(unprinted_designs[:], completed_models[:])
show_completed_models(completed_models[:])
print("传递非副本");
print_models(unprinted_designs, completed_models)
show_completed_models(completed_models)
print("传递非副本，查看变化");
print_models(unprinted_designs, completed_models)
show_completed_models(completed_models)

from learn.chapter8 import C8Import ###导入导入导入导入导入导入导入导入导入导入导入导入导入导入
C8Import.make_pizza(16, 'pepperoni')
C8Import.make_pizza(12, 'mushrooms', 'green peppers', 'extra cheese')

from learn.chapter8 import C8Import as mp ###导入导入导入导入导入导入导入导入导入导入导入导入导入导入
mp.make_pizza(16, 'pepperoni')
mp.make_pizza(12, 'mushrooms', 'green peppers', 'extra cheese')

from learn.chapter8.C8Import import * ###导入导入导入导入导入导入导入导入导入导入导入导入导入导入
make_pizza(16, 'pepperoni')
make_pizza(12, 'mushrooms', 'green peppers', 'extra cheese')
