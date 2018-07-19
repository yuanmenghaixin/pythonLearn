# reverseargs.py
import sys
args = sys.argv[1:]
args.reverse()
print(' '.join(args))
names = ['anne', 'beth', 'george', 'damon']
ages = [12, 45, 32, 102]
addresss = [12, 45, 32, 102]
for i in range(len(names)):
    print(names[i], 'is', ages[i], 'years old')
list(zip(names, ages,addresss))
for name, age,addresss in zip(names, ages,addresss):
    print(name, 'is', age, 'years old'+'addresss')