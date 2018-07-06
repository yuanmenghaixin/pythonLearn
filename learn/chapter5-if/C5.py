###########if
cars = ['audi', 'bmw', 'subaru', 'toyota']
for car in cars:
    if car == 'bmw':
        print(car.upper())
    else:
        print(car.title())

requested_topping = 'mushrooms'
if requested_topping != 'anchovies':
    print("Hold the anchovies!")

age_0 = 22
age_1 = 18
print(age_0 >= 21 and age_1 >= 21)  # and
age_1 = 22
print(age_0 >= 21 or age_1 >= 21)  # or

requested_toppings = ['mushrooms', 'onions', 'pineapple']
print('mushrooms' in requested_toppings)  # in
print('pepperoni' in requested_toppings)  # in

banned_users = ['andrew', 'carolina', 'david']
user = 'marie'
if user not in banned_users:  # not in
    print(user.title() + ", you can post a response if you wish.")

car = 'subaru'
print("Is car == 'subaru'? I predict True.")
print(car == 'subaru')
print("\nIs car == 'audi'? I predict False.")
print(car == 'audi')

age = 12
if age < 4:
    print("Your admission cost is $0.")  # fi elif else
elif age < 18:
    print("Your admission cost is $5.")
else:
    print("Your admission cost is $10.")

requested_toppings = []
if requested_toppings:
    for requested_topping in requested_toppings:
        print("Adding " + requested_topping + ".")
        print("\nFinished making your pizza!")
else:
    print("Are you sure you want a plain pizza?")


available_toppings = ['mushrooms', 'olives', 'green peppers','pepperoni', 'pineapple', 'extra cheese']
requested_toppings = ['mushrooms', 'french fries', 'extra cheese']
for requested_toppings in requested_toppings:
    if requested_topping in available_toppings:
        print("Adding " + requested_topping + ".")
    else:
        print("Sorry, we don't have " + requested_topping + ".")
print("\nFinished making your pizza!")