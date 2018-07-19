def print_params(*params):
    print(params)
print_params('Testing')
print_params(1, 2, 3)

def search(sequence, number, lower, upper):
    if lower == upper:
        assert number == sequence[upper]
        return upper
    else:
        middle = (lower + upper) // 2
        if number > sequence[middle]:
            return search(sequence, number, middle + 1, upper)
        else:
            return search(sequence, number, lower, middle)

seq = [34, 67, 8, 123, 4, 100, 95]
seq.sort()
print(seq)
print(search(seq, 34,1,5))

class Person:
    def set_name(self, name):
        self.name = name
    def get_name(self):
        return self.name
    def greet(self):
        print("Hello, world! I'm {}.".format(self.name))

foo = Person()
bar = Person()
foo.set_name('Luke Skywalker')
bar.set_name('Anakin Skywalker')
foo.greet()

class Calculator:
    def calculate(self, expression):
        self.value = eval(expression)
class Talker:
    def talk(self):
        print('Hi, my value is', self.value)
class TalkingCalculator(Calculator, Talker):
    pass

tc = TalkingCalculator()
tc.calculate('1 + 2 * 3')
tc.talk()
print(callable(getattr(tc, 'talk', None)))

try:
    print("'This's OK!")
except AttributeError:
    print('The object is not writeable')
else:
    print('The object is writeable')
# from warnings import warn
# warn("I've got a bad feeling about this.")

class FooBar:
    def __init__(self, value=42):
        self.somevar = value
f = FooBar('This is a constructor argument')
print(f.somevar)